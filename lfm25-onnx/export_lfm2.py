"""Export LiquidAI/LFM2.5-1.2B-Instruct to ONNX (prefill graph, fp32, external data).

Prefill only (`use_cache=False`): the KV/conv cache plumbing is an inference
optimisation, not architecture, and including it buries the 10-conv/6-attention
interleave under cache-shuffling nodes.  The point of this export is to make the
LFM2 hybrid legible in Netron.

Run with PYTHONUTF8=1.
"""

from __future__ import annotations

import os
import time

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "models", "LFM2.5-1.2B-Instruct")
OUT_DIR = os.path.join(ROOT, "onnx", "LFM2.5-1.2B-Instruct")
OUT = os.path.join(OUT_DIR, "model.onnx")


class PrefillWrapper(nn.Module):
    """Logits-only, cache-free view of Lfm2ForCausalLM."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        ).logits


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.manual_seed(0)

    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained(SRC)
    model = AutoModelForCausalLM.from_pretrained(SRC, dtype=torch.float32)
    model.eval()
    model.config.use_cache = False

    n = sum(p.numel() for p in model.parameters())
    types = list(cfg.layer_types)
    print(f"[ok] loaded {type(model).__name__}: {n / 1e6:.2f} M params, dtype={next(model.parameters()).dtype}")
    print(f"     layers: {len(types)} = {types.count('conv')} conv + "
          f"{types.count('full_attention')} full_attention")
    print(f"     attention at indices {[i for i, t in enumerate(types) if t == 'full_attention']}")
    ff = model.model.layers[0].feed_forward.w1.weight.shape[0]
    print(f"     feed_forward width: {ff} (config.intermediate_size={cfg.intermediate_size}, auto-adjusted)")
    # transformers 5.x renamed the LFM2 key tie_embedding -> tie_word_embeddings
    tied = getattr(cfg, "tie_word_embeddings", None)
    if tied is None:
        tied = getattr(cfg, "tie_embedding", None)
    print(f"     tied embeddings: {tied}  conv_L_cache={cfg.conv_L_cache}")

    wrapper = PrefillWrapper(model)
    S = 32
    input_ids = torch.randint(0, cfg.vocab_size, (1, S), dtype=torch.long)
    attention_mask = torch.ones(1, S, dtype=torch.long)
    args = (input_ids, attention_mask)

    with torch.no_grad():
        ref = wrapper(*args)
    print(f"[ok] torch forward: logits={tuple(ref.shape)}")

    seq = torch.export.Dim("sequence", min=2, max=4096)
    dynamic_shapes = {
        "input_ids": {1: seq},
        "attention_mask": {1: seq},
    }

    t0 = time.time()
    torch.onnx.export(
        wrapper,
        args,
        OUT,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_shapes=dynamic_shapes,
        opset_version=20,
        dynamo=True,
        external_data=True,
    )
    print(f"[ok] exported in {time.time() - t0:.1f}s")
    for p in (OUT, OUT + ".data"):
        if os.path.exists(p):
            print(f"     {os.path.basename(p):20} {os.path.getsize(p) / 1e6:10.2f} MB")

    # free the torch model before ORT loads 4.7 GB of initializers
    ref_np = ref.detach().numpy()
    del wrapper, model, ref
    import gc

    gc.collect()

    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(OUT, so, providers=["CPUExecutionProvider"])
    got = sess.run(None, {
        "input_ids": input_ids.numpy(),
        "attention_mask": attention_mask.numpy(),
    })[0]
    d = float(np.abs(got - ref_np).max())
    rel = d / max(float(np.abs(ref_np).max()), 1e-9)
    agree = float((got.argmax(-1) == ref_np.argmax(-1)).mean()) * 100
    print(f"\n[verify] logits shape={got.shape} max|diff|={d:.3e} rel={rel:.2e}")
    print(f"         argmax agreement: {agree:.2f}%")
    print("\n[done]" if d < 1e-3 else "\n[FAILED]")


if __name__ == "__main__":
    main()
