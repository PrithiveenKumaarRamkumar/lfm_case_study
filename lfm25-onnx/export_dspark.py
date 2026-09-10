"""Export the reconstructed LFM2.5-1.2B-Instruct-DSpark drafter to ONNX.

Run with PYTHONUTF8=1 (the dynamo exporter prints a check-mark emoji that a
cp1252 console cannot encode -- see logs/ and the session notes).
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
import torch

from dspark_model import load_dspark

ROOT = os.path.dirname(os.path.abspath(__file__))
DRAFT_DIR = os.path.join(ROOT, "models", "LFM2.5-1.2B-Instruct-DSpark")
TARGET_DIR = os.path.join(ROOT, "models", "LFM2.5-1.2B-Instruct")
OUT_DIR = os.path.join(ROOT, "onnx", "LFM2.5-1.2B-Instruct-DSpark")
OUT = os.path.join(OUT_DIR, "model.onnx")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    torch.manual_seed(0)

    # ---- the tied target embedding doubles as the drafter's lm_head ----
    from safetensors import safe_open

    with safe_open(os.path.join(TARGET_DIR, "model.safetensors"), framework="pt") as f:
        embed = f.get_tensor("model.embed_tokens.weight").to(torch.float32)
    print(f"[ok] target embedding {tuple(embed.shape)} "
          f"({embed.numel() / 1e6:.2f} M, tied -> also the lm_head)")

    model, cfg, n_ckpt = load_dspark(DRAFT_DIR, target_embedding=embed)
    draft_params = sum(
        p.numel() for n, p in model.named_parameters()
    )
    print(f"[ok] strict load: {n_ckpt} checkpoint tensors, 0 missing, 0 unexpected")
    print(f"     draft params       : {draft_params / 1e6:.2f} M")
    print(f"     shared target head : {embed.numel() / 1e6:.2f} M")
    print(f"     block_size={cfg['block_size']}  taps={cfg['dflash_config']['target_layer_ids']}"
          f"  mask_token_id={cfg['dflash_config']['mask_token_id']}")

    # ---- dummy inputs -------------------------------------------------
    B, T, C = 1, cfg["block_size"], 12
    V, H = cfg["vocab_size"], cfg["hidden_size"]
    F_in = model.num_context_features * H

    input_ids = torch.full((B, T), cfg["dflash_config"]["mask_token_id"], dtype=torch.long)
    positions = torch.arange(C, C + T, dtype=torch.long).unsqueeze(0).expand(B, T).contiguous()
    target_hidden = torch.randn(B, C, F_in) * 0.02
    ctx_positions = torch.arange(C, dtype=torch.long).unsqueeze(0).expand(B, C).contiguous()
    prev_token_ids = torch.randint(0, V, (B, T), dtype=torch.long)
    args = (input_ids, positions, target_hidden, ctx_positions, prev_token_ids)

    with torch.no_grad():
        ref = model(*args)
    print(f"[ok] torch forward: logits={tuple(ref[0].shape)} "
          f"hidden={tuple(ref[1].shape)} confidence={tuple(ref[2].shape)}")

    # ---- export -------------------------------------------------------
    batch = torch.export.Dim("batch", min=1, max=64)
    block = torch.export.Dim("block", min=2, max=64)
    ctx = torch.export.Dim("ctx", min=2, max=4096)
    dynamic_shapes = {
        "input_ids": {0: batch, 1: block},
        "positions": {0: batch, 1: block},
        "target_hidden": {0: batch, 1: ctx},
        "ctx_positions": {0: batch, 1: ctx},
        "prev_token_ids": {0: batch, 1: block},
    }

    t0 = time.time()
    torch.onnx.export(
        model,
        args,
        OUT,
        input_names=["input_ids", "positions", "target_hidden",
                     "ctx_positions", "prev_token_ids"],
        output_names=["logits", "hidden_states", "confidence"],
        dynamic_shapes=dynamic_shapes,
        opset_version=20,
        dynamo=True,
        external_data=True,
    )
    print(f"[ok] exported in {time.time() - t0:.1f}s")
    for p in (OUT, OUT + ".data"):
        if os.path.exists(p):
            print(f"     {os.path.basename(p):20} {os.path.getsize(p) / 1e6:10.2f} MB")

    # ---- verify against ONNX Runtime ----------------------------------
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(OUT, so, providers=["CPUExecutionProvider"])
    feed = {
        "input_ids": input_ids.numpy(),
        "positions": positions.numpy(),
        "target_hidden": target_hidden.numpy(),
        "ctx_positions": ctx_positions.numpy(),
        "prev_token_ids": prev_token_ids.numpy(),
    }
    got = sess.run(None, feed)
    print("\n[verify] ONNX Runtime vs PyTorch")
    ok = True
    for name, a, b in zip(("logits", "hidden_states", "confidence"), got, ref):
        b = b.detach().numpy()
        d = float(np.abs(a - b).max())
        rel = d / max(float(np.abs(b).max()), 1e-9)
        good = d < 1e-3
        ok &= good
        print(f"  {'OK ' if good else 'BAD'} {name:14} {str(a.shape):18} "
              f"max|diff|={d:.3e}   rel={rel:.2e}")

    # dynamic-axis smoke test: different batch / block / ctx
    feed2 = {
        "input_ids": torch.randint(0, V, (2, 5)).numpy(),
        "positions": torch.arange(20, 25).unsqueeze(0).expand(2, 5).contiguous().numpy(),
        "target_hidden": (torch.randn(2, 20, F_in) * 0.02).numpy(),
        "ctx_positions": torch.arange(20).unsqueeze(0).expand(2, 20).contiguous().numpy(),
        "prev_token_ids": torch.randint(0, V, (2, 5)).numpy(),
    }
    out2 = sess.run(None, feed2)
    print(f"  OK  dynamic axes    batch=2 block=5 ctx=20 -> logits{out2[0].shape}")
    print("\n[done]" if ok else "\n[FAILED]")


if __name__ == "__main__":
    main()
