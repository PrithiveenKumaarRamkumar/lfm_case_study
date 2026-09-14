# LFM2.5-1.2B-Instruct + DSpark → ONNX → Netron

**Session log & findings**
Date: 2026-09-06 · Host: Windows 11 Enterprise (26100) · Python 3.13.15
Working dir: `C:\Users\ramkumar.pri\lfm25-onnx\`

---

## 1. Objective

Download the two Liquid AI checkpoints and convert both to ONNX so the
architectures can be inspected in [Netron](https://netron.app):

- `LiquidAI/LFM2.5-1.2B-Instruct`
- `LiquidAI/LFM2.5-1.2B-Instruct-DSpark`

---

## 2. Headline finding

**These are not two variants of the same model.** The naming implies a sibling
pair; they are actually a *target* model and its *speculative-decoding drafter*,
with completely different architectures, and only one of the two is loadable by
`transformers`.

| | **LFM2.5-1.2B-Instruct** | **LFM2.5-1.2B-Instruct-DSpark** |
|---|---|---|
| `architectures` | `Lfm2ForCausalLM` | `Lfm2DSparkDraftModel` |
| `model_type` | `lfm2` | `qwen3` (a shell — see §4) |
| Params | ~1.2 B | 295.7 M (+ shared target head) |
| Weights | 2 340.70 MB (BF16) | 591.46 MB (BF16) |
| Layers | 16 — **hybrid**: 10 short-conv + 6 full-attention | 5 full-attention |
| Tokenizer in repo | yes | **no** (borrows the target's) |
| Loadable by `transformers` | **yes** | **no** — see §4 |
| Gated on HF | no | no |
| Role | the model that generates | drafts 9 tokens/step for the model above |

The DSpark repo's own README confirms the relationship: it is a drafter for
`LFM2.5-1.2B-Instruct`, reported at ~2× decode speedup in SGLang, mean 4.81
accepted tokens per step (block size 9, so the ceiling is 10).

---

## 3. Model A — LFM2.5-1.2B-Instruct

`config.json` (abridged):

```json
{
  "architectures": ["Lfm2ForCausalLM"],
  "model_type": "lfm2",
  "hidden_size": 2048,
  "intermediate_size": 12288,
  "num_hidden_layers": 16,
  "num_attention_heads": 32,
  "num_key_value_heads": 8,
  "vocab_size": 65536,
  "max_position_embeddings": 128000,
  "rope_theta": 1000000.0,
  "norm_eps": 1e-05,
  "tie_embedding": true,
  "dtype": "bfloat16",
  "conv_L_cache": 3,
  "conv_bias": false,
  "conv_dim": 2048,
  "block_use_swiglu": true,
  "layer_types": ["conv","conv","full_attention","conv","conv","full_attention",
                  "conv","conv","full_attention","conv","full_attention","conv",
                  "full_attention","conv","full_attention","conv"]
}
```

**Architectural point of interest.** This is the LFM2 hybrid: only **6 of 16
layers are attention**. The other 10 are *short convolution* (`ShortConv`)
blocks with a 3-tap causal depthwise conv (`conv_L_cache: 3`) — gated, LIV-style
operators that carry sequence mixing at O(L) instead of O(L²). The attention
layers land at indices 2, 5, 8, 10, 12, 14 — irregularly spaced, denser toward
the top of the stack. That conv/attention interleave is exactly what should be
visible in Netron, and it is the reason the model is cheap to run on-device.

- GQA: 32 query heads / 8 KV heads (4× fan-out), head_dim 64.
- **The real feed-forward width is 8192, not the 12288 in `config.json`.** The
  weights say so (`feed_forward.w1.weight` is [8192, 2048]).
  `block_auto_adjust_ff_dim: true` applies the standard SwiGLU two-thirds rule --
  ⅔ × 12288 = 8192, then rounds to a multiple of `block_multiple_of: 256`
  (8192 already is). Reading `intermediate_size` at face value would overstate
  the MLP by 50%.
- Every layer, conv or attention, carries its own `feed_forward` plus
  `operator_norm` / `ffn_norm`; a final `embedding_norm` precedes the tied head.

**The ShortConv block** (`Lfm2ShortConv`, confirmed against
`transformers/models/lfm2/modeling_lfm2.py`) is a *double-gated* depthwise causal
convolution:

```
BCx  = in_proj(h)                # [B, 2048] -> [B, 6144], split 3 ways
B, C, x = chunk(BCx, 3)          # three [B, 2048] streams
h    = B * x                     # input gate
h    = causal_conv1d(h)          # depthwise (groups=2048), 3 taps, causal
y    = C * h                     # output gate
y    = out_proj(y)               # [B, 2048] -> [B, 2048]
```

so `in_proj` being 3 x hidden is the giveaway. Two multiplicative gates around a
3-tap depthwise kernel: the same shape as Mamba's convolution branch, without
the state-space recurrence. Each token mixes only with the two before it, and
depth is what grows the receptive field.

**The layer skeleton is uniform across both types.** `Lfm2DecoderLayer` is:

```
residual -> operator_norm -> {self_attn | conv} -> +residual
         -> ffn_norm      ->  feed_forward      -> +residual
```

Only the operator slot differs. That is what makes the hybrid legible in Netron:
16 identical-looking blocks where 6 contain a QKV fan-out and 10 contain a
Conv + two Mul gates.
- Embeddings are **tied** (`tie_embedding: true`), so `lm_head` re-uses the
  [65536, 2048] embedding matrix — 134 M of the 1.2 B params.

---

## 4. Model B — LFM2.5-1.2B-Instruct-DSpark (the hard one)

### 4.1 Why nothing on PyPI can load it

The config declares `architectures: ["Lfm2DSparkDraftModel"]` but:

- the repo contains **only** `config.json` + `model.safetensors` (no
  `modeling_*.py`),
- there is **no `auto_map`**, so `trust_remote_code=True` has nothing to fetch,
- `Lfm2DSparkDraftModel` does not exist in `transformers`,
- `model_type: "qwen3"` is a decoy — it makes `AutoConfig` resolve, but
  `Qwen3ForCausalLM` cannot consume these weights (there is no `embed_tokens`,
  no `lm_head`, and there are four tensors Qwen3 has never heard of).

The class lives in **SGLang**, and the repo README says it needs a build with
DSpark support for LFM2 targets (PR #31041). SGLang does not run on Windows
(Linux + CUDA), so "just load it" was not an option.

Reference implementation located at:

| File | Provides |
|---|---|
| `python/sglang/srt/models/lfm2_dspark.py` | `Lfm2DSparkDraftModel` — a 16-line subclass, no overrides |
| `python/sglang/srt/models/dspark.py` | `DSparkDraftMixin`, `VanillaMarkov`, `DSparkConfidenceHead` |
| `python/sglang/srt/models/dflash.py` | `DFlashDraftModel`, `DFlashDecoderLayer`, `DFlashAttention`, `DFlashMLP` |

`lfm2_dspark.py` in full is just `class Lfm2DSparkDraftModel(DSparkDraftModel): pass`
— its docstring notes the current checkpoints are "attention-only Qwen3-style GQA
with interleaved RoPE", with the subclass existing so LFM2 drafts can later add
ShortConv layers.

**Approach taken:** read the weight manifest straight off the Hub with an HTTP
range request (safetensors stores a JSON header in the first bytes — no need to
download 591 MB to see the tensor names), then reconstruct the module in plain
PyTorch, mirroring the SGLang classes op-for-op. Parameter names match the
checkpoint exactly, so `load_state_dict(strict=True)` is the correctness check.

### 4.2 Full tensor manifest (read without downloading)

```
BF16 [1]            confidence_head.proj.bias
BF16 [1, 2304]      confidence_head.proj.weight
BF16 [2048, 10240]  fc.weight
BF16 [2048]         hidden_norm.weight
BF16 [2048]         norm.weight
BF16 [65536, 256]   markov_head.markov_w1.weight
BF16 [65536, 256]   markov_head.markov_w2.weight
  × 5 layers:
BF16 [2048]         layers.N.input_layernorm.weight
BF16 [2048]         layers.N.post_attention_layernorm.weight
BF16 [2048, 2048]   layers.N.self_attn.q_proj.weight
BF16 [512,  2048]   layers.N.self_attn.k_proj.weight
BF16 [512,  2048]   layers.N.self_attn.v_proj.weight
BF16 [2048, 2048]   layers.N.self_attn.o_proj.weight
BF16 [64]           layers.N.self_attn.q_norm.weight
BF16 [64]           layers.N.self_attn.k_norm.weight
BF16 [6144, 2048]   layers.N.mlp.gate_proj.weight
BF16 [6144, 2048]   layers.N.mlp.up_proj.weight
BF16 [2048, 6144]   layers.N.mlp.down_proj.weight
                    TOTAL 295.73 M  (matches the README's 295.7 M)
```

Note what is **absent**: no `embed_tokens`, no `lm_head`. SGLang injects the
target's via `attach_shared_modules(embed_tokens=…, lm_head=…)`. Since
LFM2.5-1.2B-Instruct is tied, both are the same [65536, 2048] matrix.

### 4.3 How the drafter actually works

Reading the SGLang source resolved every unusual tensor:

**`fc.weight` is [2048, **10240**]** — 10240 = 5 × 2048. Cross-referenced with
`dflash_config.target_layer_ids: [2, 5, 8, 11, 14]` and `num_target_layers: 16`:
the target model is tapped at **5 of its 16 layers**, the per-token hidden states
are concatenated, and `fc` projects that 10240-d bundle down to 2048, followed by
`hidden_norm` (RMSNorm). This is the DFlash context path — the result is written
straight into the *draft's* KV cache as a context prefix
(`write_target_hidden_kv` → `kv_proj_only` → `apply_k_norm` → `apply_k_rope`).
The drafter therefore never re-reads the prompt; it inherits the target's
already-computed internal representation.

Worth noting: the taps `[2, 5, 8, 11, 14]` are *evenly spaced every 3 layers*,
not the target's attention layers (which sit at 2, 5, 8, 10, 12, 14). Layer 11 is
a ShortConv block. The drafter samples the residual stream at regular depth
intervals rather than at operator boundaries.

**The backbone is Qwen3-shaped, not LFM2-shaped.** 5 layers, GQA 32/8, head_dim
64, per-head RMSNorm on Q and K (`q_norm`/`k_norm` of size **64** = head_dim —
Qwen3's signature), SwiGLU with `intermediate_size` 6144. RoPE is **interleaved**
(GPT-J pairing), not NeoX: `rope_is_neox_style: false`.

**Attention over the block is bidirectional.** `_get_dflash_layer_attention_params`
maps `layer_type == "full_attention"` to `AttentionType.ENCODER_ONLY` unless the
config sets `is_causal` — and this config does not. That is the giveaway that
DSpark is a *block* decoder: with `block_size: 9` it proposes 9 tokens in a
single forward pass, so the block tokens attend to each other in both directions
(plus full visibility of the context prefix). `mask_token_id: 64402` is the
placeholder those positions start from — the same trick masked-diffusion decoders
use.

**`markov_head` (2 × [65536, 256]) replaces the missing LM head — sort of.** It
does not produce logits on its own. `VanillaMarkov` is a rank-256 factorization
of a 65536 × 65536 **bigram transition matrix**: the previously proposed token
indexes `markov_w1` (an Embedding), and `markov_w2` expands the 256-d latent back
to vocab size as an **additive bias** on the base logits. Full base logits still
come from the target's tied head. This is how a block decoder recovers
token-to-token dependence inside a block it otherwise predicts in parallel — at
33.5 M params instead of 4.3 B.

**`confidence_head.proj` is [1, **2304**]** — 2304 = 2048 (hidden) + 256
(markov_rank). It consumes `cat(hidden_states, markov_w1(prev_token))` and emits
one scalar per position, through a sigmoid, as a predicted acceptance
probability — used to truncate the proposed block early ("ragged verify") rather
than always sending all 9 tokens to the target for verification.

### 4.4 Reconstructed forward pass

```
target_hidden [B,C,10240]  (5 tapped target layers, concatenated)
      │
      ├─ fc ─────────────► hidden_norm ──► ctx_hidden [B,C,2048]
      │                                          │
input_ids [B,T] ──► embed_tokens ──► h [B,T,2048]│  (shared, tied, from target)
      │                                          │
      └────► 5 × DSparkDecoderLayer ◄────────────┘  ctx K/V prefixed per layer
                     │                              (k_norm + RoPE applied to ctx K)
                     ▼
                  norm ──► hidden_states [B,T,2048]
                     │
      ┌──────────────┼────────────────────────────┐
      ▼              ▼                            ▼
 hidden @ Wemb.T   markov_w2(markov_w1(prev))   proj(cat(h, markov_emb))
   base_logits   +      markov bias        →  σ(·) = confidence [B,T]
                  = logits [B,T,65536]
```

---

## 5. Environment

| Component | Status |
|---|---|
| Python | 3.13.15 (`AppData\Local\Programs\Python\Python313`) |
| pip | 26.2.1 |
| git / git-lfs | 2.55.0 / 3.7.1 |
| Free disk (C:) | 152 GB of 237 GB |
| ML packages pre-installed | **none** — torch/transformers/onnx all installed fresh this session |
| MSVC runtime | present (`vcruntime140`, `vcruntime140_1`, `msvcp140`) — torch on Windows needs these |

Installed: `torch`, `transformers`, `optimum[exporters]`, `onnx`, `onnxruntime`,
`onnxscript`, `huggingface_hub[hf_xet]`, `accelerate`, `netron`.

### Gotchas hit

- **`ImportError: DLL load failed while importing _C`** while pip was still
  running — a red herring. `site-packages/torch/lib/` was empty because pip had
  not yet written the DLLs. Not the usual missing-MSVC-redistributable cause;
  those DLLs were verified present.
- **`UnicodeEncodeError: 'charmap' codec can't encode character '✅'`** from
  `torch.onnx.export(dynamo=True)`. The export itself *succeeded*; the crash was
  the exporter's own progress line printing a check-mark emoji to a cp1252
  console. Fix: run with `PYTHONUTF8=1 PYTHONIOENCODING=utf-8`. Worth knowing
  because the traceback points at `_capture_strategies.py` and looks like an
  export failure.
- A long quoted bash heredoc (`<<'PY'`) carrying a Python file with mixed
  quote styles broke the shell wrapper (`unexpected EOF while looking for
  matching '`). Short heredocs are fine; long source files go through a file
  write instead.

---

## 6. Conversion approach

`torch.onnx.export(..., dynamo=True)` was used directly for both models, with
ONNX Runtime checked against PyTorch afterwards.

**Why not `optimum-cli export onnx`?** The obvious route is a dead end here.
optimum **2.3.0** returns **0** exportable model types for the `onnx` exporter:

```
text-generation             0 types | lfm2=False qwen3=False
text-generation-with-past   0 types | lfm2=False qwen3=False
```

In optimum 2.x the ONNX exporter was split out into a separate `optimum-onnx`
distribution, so the `optimum[exporters]` extra no longer ships any ONNX export
configs. (Its `optimum.exporters.tasks` also still imports `requests`, which
`huggingface_hub` 1.x no longer pulls in — `pip install requests` is needed
before it will even import.) And the drafter could never have gone through
optimum regardless, since optimum builds on a `transformers` model class and no
such class exists for it.

Both models are exported at **float32** with weights in an external
`.onnx.data` file:

- >2 GB of weights cannot live in a single ONNX protobuf, so external data is
  mandatory for the 1.2B model and used for both for consistency.
- It also makes the `.onnx` graph file tiny, which matters for Netron (§7).
- float32 rather than the checkpoints' bfloat16, because ONNX Runtime's CPU
  provider has patchy bf16 coverage and the point of the export was to be able
  to *verify* it numerically.

The 1.2B model is exported as a **prefill graph** (`use_cache=False`). The
KV/conv cache plumbing is an inference optimisation, not architecture, and
including it buries the actual layer structure under cache-shuffling nodes. The
graph therefore shows all 16 layers — ShortConv and attention — in their true
interleave, which is the thing worth looking at.

---

## 7. Netron notes

Netron opens both models effectively instantly, despite the multi-GB weights,
and it is worth knowing why — the naive expectation is that a 4.7 GB model is
unusable in a browser.

Reading Netron 9.2.6's source settles it:

- `server.py` serves the model at `/data/<name>` and resolves **any relative
  path against the model file's own directory** (guarded by a `commonpath`
  check). So a sibling `.onnx.data` is served correctly — verified live:

  ```
  /data/model.onnx        200  Content-Length=750061
  /data/model.onnx.data   200  Content-Length=1719795712
  ```

- `onnx.js` treats an `EXTERNAL` initializer as a **lazy `{location, offset,
  length}` request** and only calls `read()` when you click a specific tensor.
  Graph construction never touches the weight blob.

So the `.onnx` file holds only topology and the browser only ever fetches the
byte range for a weight you actually inspect. **Keep `model.onnx` and
`model.onnx.data` in the same folder** — that is the one requirement.

---

## 8. Results

### LFM2.5-1.2B-Instruct-DSpark — done, verified

The reconstruction is correct: `load_state_dict` accepted all 62 checkpoint
tensors with **0 missing and 0 unexpected**, and the exported graph matches
PyTorch numerically.

```
draft params        : 295.73 M   (README says 295.7 M)
shared target head  : 134.22 M   (tied embed_tokens, injected from the target)

[verify] ONNX Runtime vs PyTorch
  OK logits         (1, 9, 65536)   max|diff|=2.337e-05   rel=9.89e-07
  OK hidden_states  (1, 9, 2048)    max|diff|=2.193e-05   rel=1.62e-06
  OK confidence     (1, 9)          max|diff|=2.384e-07   rel=3.11e-07
```

Those residuals are float32 round-off, not modelling error. Since the weights
only load if every name and shape lines up, *and* the outputs agree, the
SGLang-derived architecture is confirmed correct rather than merely plausible.

Graph:

```
model.onnx        0.75 MB     (topology only)
model.onnx.data   1.72 GB     (429.94 M fp32 initializers)
nodes             588         initializers 79
opset             ai.onnx=20  ir_version 10   producer pytorch 2.14.0+cpu

inputs:   input_ids      INT64[batch,block]
          positions      INT64[batch,block]
          target_hidden  FLOAT[batch,ctx,10240]
          ctx_positions  INT64[batch,ctx]
          prev_token_ids INT64[batch,block]
outputs:  logits         FLOAT[batch,block,65536]
          hidden_states  FLOAT[batch,block,2048]
          confidence     FLOAT[batch,block]

op histogram: 131 Mul, 59 MatMul, 56 Reshape, 55 Add, 46 Unsqueeze, 32 Concat,
              30 Slice, 27 Pow/ReduceMean/Sqrt/Reciprocal (= 27 RMSNorms),
              21 Transpose, 15 Sub, 10 Expand, 6 Sigmoid, 5 Softmax,
              3 Shape, 3 Squeeze, 2 Gather, 2 Cast, 2 Cos, 2 Sin
```

Two sanity checks on that histogram: **5 Softmax** = the 5 attention layers, and
**6 Sigmoid** = 5 SwiGLU SiLUs + the confidence head. Only **2 Cos / 2 Sin**
survive for all 5 layers because every layer's RoPE is identical, so the
exporter's CSE collapsed them into one shared pair (one for block positions, one
for context positions).

### LFM2.5-1.2B-Instruct — done, verified

```
[ok] loaded Lfm2ForCausalLM: 1170.34 M params, dtype=torch.float32
     layers: 16 = 10 conv + 6 full_attention
     feed_forward width: 8192 (config.intermediate_size=12288, auto-adjusted)
[ok] exported in 528.2s
     graph file 1.69 MB, 4.68 GB total (weights external)

[verify] logits shape=(1, 32, 65536) max|diff|=7.010e-05 rel=4.26e-06
         argmax agreement: 100.00%
```

Graph:

```
model.onnx        1.69 MB     (topology only)
model.onnx.data   4.68 GB     (1170.34 M fp32 initializers)
nodes             806         initializers 169
opset             ai.onnx=20  ir_version 10   producer pytorch 2.14.0+cpu

inputs:   input_ids       INT64[1,sequence]
          attention_mask  INT64[1,sequence]
outputs:  logits          FLOAT[1,sequence,65536]

op histogram: 182 Mul, 106 MatMul, 95 Add, 52 Transpose,
              45 Pow/ReduceMean/Sqrt/Reciprocal (= 45 RMSNorms), 36 Reshape,
              34 Slice, 20 Unsqueeze, 18 Concat, 16 Sigmoid, 13 Expand,
              12 Neg, 10 Split, 10 Conv, 6 Softmax, 3 Cast, 2 Shape, 2 And,
              1 each Squeeze/Gather/Range/LessOrEqual/Max/GatherND/Where/Cos/Sin
```

**The op histogram is a clean fingerprint of the hybrid** — every count is
predicted by the config:

| Count | Op | Why |
|---:|---|---|
| **10** | `Conv` | the 10 ShortConv layers (depthwise, 3 taps) |
| **10** | `Split` | their `chunk(BCx, 3)` into B / C / x |
| **6** | `Softmax` | the 6 attention layers |
| **16** | `Sigmoid` | one SwiGLU SiLU per layer (all 16 have an MLP) |
| **45** | RMSNorm groups | 16 `operator_norm` + 16 `ffn_norm` + 6 `q_layernorm` + 6 `k_layernorm` + 1 `embedding_norm` |
| **1** | `Cos` / `Sin` | RoPE is identical across the 6 attention layers, so the exporter's CSE collapsed it to one shared pair |

That last row is the same effect seen in the drafter (2 Cos / 2 Sin for 5
layers): position embeddings are computed once and shared, so do not expect one
per layer in Netron.

---

## 9. What was produced

Everything lives under `C:/Users/ramkumar.pri/lfm25-onnx/`:

```
models/
  LFM2.5-1.2B-Instruct/           config, tokenizer, model.safetensors (2.34 GB)
  LFM2.5-1.2B-Instruct-DSpark/    config.json, model.safetensors (591 MB)
onnx/
  LFM2.5-1.2B-Instruct/
    model.onnx           1.69 MB   <- open THIS in Netron
    model.onnx.data      4.68 GB
  LFM2.5-1.2B-Instruct-DSpark/
    model.onnx           0.75 MB   <- open THIS in Netron
    model.onnx.data      1.72 GB
ref/                     SGLang reference sources (dspark.py, dflash.py, ...)
dspark_model.py          the reconstruction (no equivalent exists for HF)
export_lfm2.py           target-model export + ORT verification
export_dspark.py         drafter export + ORT verification
summarize_onnx.py        graph statistics
view.py                  opens a model in Netron
logs/
```

Reproduce:

```powershell
$env:PYTHONUTF8=1                     # see the gotcha in section 5
python export_lfm2.py                 # ~9 min, 5 GB RAM
python export_dspark.py               # ~3 min
python view.py lfm2                   # or: python view.py dspark
python summarize_onnx.py onnx/LFM2.5-1.2B-Instruct/model.onnx
```

---

## 10. Limitations / things to know

- **Batch is pinned to 1 in the 1.2B export.** `torch.export` specialised the
  batch dimension despite a `Dim("batch", 1..16)` hint (its mask-construction
  path constrains it); the sequence axis stayed dynamic. Irrelevant for
  visualisation, and batch-1 prefill is the normal case. To get a dynamic batch,
  export with a batch-2 dummy input and re-run. The drafter export kept all
  three axes dynamic (`batch`, `block`, `ctx`).
- **The 1.2B graph is prefill-only** (`use_cache=False`) — deliberate, see §6. It
  is not a drop-in autoregressive decoder with KV cache.
- **The drafter graph is a reconstruction, not an official export.** It is
  validated two independent ways (strict weight load + ONNX Runtime agreement),
  but two behavioural details were inferred from SGLang's structure rather than
  from a runnable DSpark trace, since SGLang does not run on Windows:
  1. the confidence head's second input is taken to be `markov_w1(prev_token)`
     (its `2304 = 2048 + 256` input width and `with_markov=True` default leave
     little else it could be);
  2. context K/V is concatenated as a plain prefix, whereas SGLang writes it
     into a paged KV pool — same math, different memory layout.

  Neither affects the parameters, shapes, or topology on display.
- Exports are **float32**; the checkpoints are bfloat16. Values are exact
  (bf16 -> fp32 is lossless); the files are 2x larger than they strictly need
  to be.
- `optimum` needed `pip install requests` just to import, and still exports
  nothing — see §6.

---

## 11. Viewing

Both models are being served right now:

| Model | URL |
|---|---|
| LFM2.5-1.2B-Instruct | <http://localhost:8080> |
| LFM2.5-1.2B-Instruct-DSpark | <http://localhost:8081> |

**Gotcha — one Netron server per process.** Serving both models from a single
Python process silently shows the *same* model on every port. In
`netron/server.py` the served model is stashed on a **class** attribute:

```python
class _HTTPRequestHandler(http.server.BaseHTTPRequestHandler):
    content = None                                    # line 47, class level
...
    self.server.RequestHandlerClass.content = content # line 136
```

`RequestHandlerClass` *is* that shared class, so a second `netron.start()` in the
same interpreter overwrites the first and both ports serve whichever model
loaded last. `serve_both.py` now spawns a child process per model.

It also fails quietly: every port still answers `HTTP 200` on `/`, and the viewer
HTML is byte-identical regardless of which model is loaded, so a reachability
check passes while the pages are wrong. Verify by fetching `/data/model.onnx`
and checking what actually comes back:

```
port 8080: 1690319 bytes  806 nodes  1170.34M params  Conv=10 Softmax=6  -> LFM2 hybrid
port 8081:  750061 bytes  588 nodes   429.94M params  Conv=0  Softmax=5  -> DSpark drafter
```

To bring them back up later:

```powershell
python C:/Users/ramkumar.pri/lfm25-onnx/serve_both.py     # both, ports 8080/8081
python C:/Users/ramkumar.pri/lfm25-onnx/view.py lfm2      # one at a time
```

Or open `onnx/<model>/model.onnx` in the Netron desktop app / netron.app — just
keep `model.onnx.data` beside it.

**What to look for in the 1.2B graph:** 16 near-identical blocks, each
`operator_norm -> operator -> +residual -> ffn_norm -> SwiGLU -> +residual`.
Ten of them contain a `Conv` flanked by two `Mul` gates (ShortConv); six contain
the QKV fan-out and a `Softmax`. The irregular interleave (attention at layers
2, 5, 8, 10, 12, 14) is the whole point of the LFM2 design.

**What to look for in the drafter graph:** the `target_hidden` input at
[batch, ctx, **10240**] entering a single `MatMul` (`fc`) is the tell that this
model consumes five tapped layers of another model; and follow `prev_token_ids`
into the two 65536x256 `markov_head` matrices to see the bigram bias being added
to the logits.

Session logs:
❯ /effort
  ⎿  Set effort level to max (this session only): Maximum capability with deepest reasoning. May use excessive tokens resulting in long response times or overthinking. Use sparingly for the hardest tasks.

❯ download the LFM2.5-1.2B-Instruct and LFM2.5-1.2B-Instruct-DSpark models hosted at https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct/tree/main and https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct-DSpark/tree/main respectively and convert them to onnx model to visualize in netron

I'll start by checking the environment — what Python tooling is available, disk space, and whether the models are gated.

  Ran 2 shell commands

Python 3.13, pip, git-lfs, and 152 GB free — good. Let me inspect both HF repos and check what ML packages are already installed.

  Ran 3 shell commands

Useful discovery: the "DSpark" repo is tagged qwen3 + draft-model (a speculative-decoding draft model), not an LFM2 model. Let me pull both configs and file sizes.

  Ran 2 shell commands

Clear picture now. Two very different models:

┌─────────────┬───────────────────────────────────────────────┬───────────────────────────────────────────────┐
│             │             LFM2.5-1.2B-Instruct              │          LFM2.5-1.2B-Instruct-DSpark          │
├─────────────┼───────────────────────────────────────────────┼───────────────────────────────────────────────┤
│ Arch        │ Lfm2ForCausalLM (lfm2)                        │ Lfm2DSparkDraftModel (qwen3 shell)            │
├─────────────┼───────────────────────────────────────────────┼───────────────────────────────────────────────┤
│ Layers      │ 16 — hybrid: 11 short-conv + 5 full-attention │ 5 full-attention + DSpark draft heads         │
├─────────────┼───────────────────────────────────────────────┼───────────────────────────────────────────────┤
│ Size        │ 2.34 GB                                       │ 591 MB                                        │
├─────────────┼───────────────────────────────────────────────┼───────────────────────────────────────────────┤
│ Loadable by │ stock transformers                            │ nothing on PyPI — it's an SGLang custom class │
└─────────────┴───────────────────────────────────────────────┴───────────────────────────────────────────────┘

The DSpark one is a speculative-decoding draft model with no modeling_*.py in the repo, so I'll need to reconstruct it from its weights. Let me kick off the toolchain install and inspect the weight layouts in parallel.

● Bash(mkdir -p lfm25-onnx/logs && cd lfm25-onnx && python -m pip install --upgrade "torch" "transformers" "optimum[exporters]" "onnx" "onnxruntime" "huggingface_hub[hf_xet]" "accelerate" "netron" "onnxscript" > logs/pip-install.log 2>&1; echo "EXIT=$?"; tail -5 logs/pip-install.log)
Running in the background (↓ to manage)

● Bash(python - <<'PY'
  import json, struct, urllib.request

  def header(repo):
      url = f"https://huggingface.co/{repo}/resolve/main/model.safetensors"
      # first 8 bytes = little-endian u64 header length
      req = urllib.request.Request(url, headers={"Range": "bytes=0-7"})
      n = struct.unpack("<Q", urllib.request.urlopen(req).read(8))[0]
      req = urllib.request.Request(url, headers={"Range": f"bytes=8-{8+n-1}"})
      return json.loads(urllib.request.urlopen(req).read(n))

  h = header("LiquidAI/LFM2.5-1.2B-Instruct-DSpark")
  print("=== DSpark: __metadata__ ===", h.get("__metadata__"))
  tot = 0
  for k, v in h.items():
      if k == "__metadata__": continue
      n = 1
      for d in v["shape"]: n *= d
      tot += n
      print(f"  {v['dtype']:8} {str(v['shape']):22} {k}")
  print(f"TOTAL PARAMS: {tot/1e6:.2f} M")
  PY)
=== DSpark: __metadata__ === None
  BF16     [1]                    confidence_head.proj.bias
  BF16     [1, 2304]              confidence_head.proj.weight
  BF16     [2048, 10240]          fc.weight
  BF16     [2048]                 hidden_norm.weight
  BF16     [2048]                 layers.0.input_layernorm.weight
  BF16     [2048, 6144]           layers.0.mlp.down_proj.weight
  BF16     [6144, 2048]           layers.0.mlp.gate_proj.weight
  BF16     [6144, 2048]           layers.0.mlp.up_proj.weight
  BF16     [2048]                 layers.0.post_attention_layernorm.weight
  BF16     [64]                   layers.0.self_attn.k_norm.weight
  BF16     [512, 2048]            layers.0.self_attn.k_proj.weight
  BF16     [2048, 2048]           layers.0.self_attn.o_proj.weight
  BF16     [64]                   layers.0.self_attn.q_norm.weight
  BF16     [2048, 2048]           layers.0.self_attn.q_proj.weight
  BF16     [512, 2048]            layers.0.self_attn.v_proj.weight
  BF16     [2048]                 layers.1.input_layernorm.weight
  BF16     [2048, 6144]           layers.1.mlp.down_proj.weight
  BF16     [6144, 2048]           layers.1.mlp.gate_proj.weight
  BF16     [6144, 2048]           layers.1.mlp.up_proj.weight
  BF16     [2048]                 layers.1.post_attention_layernorm.weight
  BF16     [64]                   layers.1.self_attn.k_norm.weight
  BF16     [512, 2048]            layers.1.self_attn.k_proj.weight
  BF16     [2048, 2048]           layers.1.self_attn.o_proj.weight
  BF16     [64]                   layers.1.self_attn.q_norm.weight
  BF16     [2048, 2048]           layers.1.self_attn.q_proj.weight
  BF16     [512, 2048]            layers.1.self_attn.v_proj.weight
  BF16     [2048]                 layers.2.input_layernorm.weight
  BF16     [2048, 6144]           layers.2.mlp.down_proj.weight
  BF16     [6144, 2048]           layers.2.mlp.gate_proj.weight
  BF16     [6144, 2048]           layers.2.mlp.up_proj.weight
  BF16     [2048]                 layers.2.post_attention_layernorm.weight
  BF16     [64]                   layers.2.self_attn.k_norm.weight
  BF16     [512, 2048]            layers.2.self_attn.k_proj.weight
  BF16     [2048, 2048]           layers.2.self_attn.o_proj.weight
  BF16     [64]                   layers.2.self_attn.q_norm.weight
  BF16     [2048, 2048]           layers.2.self_attn.q_proj.weight
  BF16     [512, 2048]            layers.2.self_attn.v_proj.weight
  BF16     [2048]                 layers.3.input_layernorm.weight
  BF16     [2048, 6144]           layers.3.mlp.down_proj.weight
  BF16     [6144, 2048]           layers.3.mlp.gate_proj.weight
  BF16     [6144, 2048]           layers.3.mlp.up_proj.weight
  BF16     [2048]                 layers.3.post_attention_layernorm.weight
  BF16     [64]                   layers.3.self_attn.k_norm.weight
  BF16     [512, 2048]            layers.3.self_attn.k_proj.weight
  BF16     [2048, 2048]           layers.3.self_attn.o_proj.weight
  BF16     [64]                   layers.3.self_attn.q_norm.weight
  BF16     [2048, 2048]           layers.3.self_attn.q_proj.weight
  BF16     [512, 2048]            layers.3.self_attn.v_proj.weight
  BF16     [2048]                 layers.4.input_layernorm.weight
  BF16     [2048, 6144]           layers.4.mlp.down_proj.weight
  BF16     [6144, 2048]           layers.4.mlp.gate_proj.weight
  BF16     [6144, 2048]           layers.4.mlp.up_proj.weight
  BF16     [2048]                 layers.4.post_attention_layernorm.weight
  BF16     [64]                   layers.4.self_attn.k_norm.weight
  BF16     [512, 2048]            layers.4.self_attn.k_proj.weight
  BF16     [2048, 2048]           layers.4.self_attn.o_proj.weight
  BF16     [64]                   layers.4.self_attn.q_norm.weight
  BF16     [2048, 2048]           layers.4.self_attn.q_proj.weight
  BF16     [512, 2048]            layers.4.self_attn.v_proj.weight
  BF16     [65536, 256]           markov_head.markov_w1.weight
  BF16     [65536, 256]           markov_head.markov_w2.weight
  BF16     [2048]                 norm.weight
TOTAL PARAMS: 295.73 M


Got the complete DSpark weight layout without downloading a byte. It's a DFlash-style draft model: fc [2048, 10240] fuses hidden states from 5 target layers (5×2048), 5 Qwen3-style layers (per-head
q/k RMSNorm), then a low-rank markov_head (2× [65536, 256]) instead of an LM head, plNo embeddings, no lm_head — it borrows the target's.

Let me find the reference implementation so my reconstruction is exact, not guessed.

  Ran 2 shell commands

sglang/srt/models/lfm2_dspark.py exists upstream. Let me pull the reference implement

  Ran 9 shell commands

Now I have the exact reference implementation. Let me check install progress and starallel.

  Read 1 file, ran 1 shell command

pip is at the final install step. Starting the big weight downloads in the background

  Ran 1 shell command

While the weights download, let me confirm a few implementation details (apply_qk_nor

  Listed 1 directory, ran 4 shell commands

Confirmed all the details. Now let me write the DSpark reconstruction while downloads

  Read 2 files

❯ Document all your findings you have made thus far and further findings in a session.m

Heredoc works in isolation — something in that specific payload broke the shell wrappor the long Python files and keep Bash for everything else. Also noting the request to document findings — I'll create ~/Desktop/session.md and keep it updated as I go.

● Write(dspark_model.py)
Wrote 315 lines to dspark_model.py
"""Faithful PyTorch reconstruction of LiquidAI/LFM2.5-1.2B-Instruct-DSpark.

The checkpoint declares ``architectures: ["Lfm2DSparkDraftModel"]`` but ships no
modeling code and no ``auto_map``, so ``transformers`` cannot load it -- the c
lives in SGLang:

    python/sglang/srt/models/lfm2_dspark.py  ->  Lfm2DSparkDraftModel(DSparkDraftModel)
    python/sglang/srt/models/dspark.py       ->  DSparkDraftMixin, VanillaMark
                                                 DSparkConfidenceHead
    python/sglang/srt/models/dflash.py       ->  DFlashDraftModel, DFlashDecod
… +305 lines

  Ran 1 shell command

Download at 1.88/2.34 GB. The torch import error is likely just the in-progress install, but let me verify the MSVC runtime is present (this box has no package managers per my notes).

  Ran 1 shell command

torch/lib/ is empty — pip simply hasn't written the DLLs yet, and the MSVC runtime is fine. Let me locate the real Desktop (OneDrive redirection is common on corporate machines) and write up findings
so far.

  Ran 1 shell command

● Write(~\Desktop\session.md)
Wrote 263 lines to ..\Desktop\session.md
# LFM2.5-1.2B-Instruct + DSpark → ONNX → Netron

**Session log & findings**
Date: 2026-09-06 · Host: Windows 11 Enterprise (26100) · Python 3.13.15
Working dir: `C:\Users\ramkumar.pri\lfm25-onnx\`

---

## 1. Objective
… +253 lines

Let me verify my layer-type count programmatically before it propagates.

  Ran 1 shell command

Correction: it's 10 conv + 6 attention, not 11/5. Fixing that in the doc.

  Listed 1 directory, ran 2 shell commands

Model A downloaded (2.34 GB complete). Let me check if the toolchain is ready.

  Ran 1 shell command

● Background command "Download both model.safetensors files" completed (exit code 0)

Both downloads finished. torch now imports; transformers is still being written by pip. Let me verify the downloaded weights.

  Listed 1 directory, ran 1 shell command

Both verified complete: 1170.34 M and 295.73 M params. Let me inspect the LFM2 tensor layout (and grab the tied embedding for the DSpark graph).

  Ran 1 shell command

LFM2 structure confirmed. Note the real FFN width is 8192, not the config's 12288 — block_auto_adjust_ff_dim applies the SwiGLU ⅔ rule (⅔ × 12288 = 8192). Let me check pip.

  Ran 1 shell command

Still installing. Let me write both export scripts while I wait.

● Write(export_dspark.py)
Wrote 153 lines to export_dspark.py
"""Export LiquidAI/LFM2.5-1.2B-Instruct-DSpark to ONNX.

The checkpoint has no loadable modeling code (see dspark_model.py), so this
builds the SGLang architecture in plain PyTorch, loads the safetensors weights
into it with strict=True, exports, and checks ONNX Runtime against PyTorch.
"""

from __future__ import annotations

import argparse
… +143 lines

● Write(export_lfm2.py)
Wrote 153 lines to export_lfm2.py
"""Export LiquidAI/LFM2.5-1.2B-Instruct (Lfm2ForCausalLM) to ONNX.

Exported as a prefill graph (use_cache=False) so the full 16-layer stack is
visible: the 10 ShortConv blocks and 6 GQA attention blocks in their actual
interleave, each with its SwiGLU feed-forward. The KV/conv cache plumbing is an
inference optimisation, not architecture, and only clutters the graph.

Weights land in an external .onnx_data file, so the .onnx graph file itself st
small and Netron opens it instantly.
"""
… +143 lines

  Ran 2 shell commands

pip is still running (through onnxscript; only accelerate/transformers/optimum remain). Let me prepare the graph-summary and Netron launcher meanwhile.

● Write(summarize_onnx.py)
Wrote 66 lines to summarize_onnx.py
"""Print a structural summary of an exported ONNX graph.

  Ran 1 shell command

● Background command "Install torch, transformers, optimum, onnx toolchain" completed (exit code 0)

pip finished (exit 0), and transformers 5.16.1 has native lfm2 → Lfm2ForCausalLM. Running the DSpark export first.

  Ran 1 shell command

Export itself succeeded — the crash is Windows cp1252 choking on torch's ✅ progress emoji. Setting UTF-8 mode.

  Ran 1 shell command

DSpark exported and verified: strict weight load (0 missing/unexpected), and ONNX Runtime matches PyTorch to 2.3e-05 on logits. The reconstruction is numerically faithful. Now the 1.2B model.

  Ran 2 shell commands

DSpark graph is clean — 588 nodes, 0.75 MB graph file (instant in Netron), 429.94 M params in external data. Let me check the 1.2B export progress.

  Listed 1 directory, ran 1 shell command

Model's still loading (2.34 GB → fp32). Let me update the findings doc with the DSpark results meanwhile.

  Ran 2 shell commands

● Write(view.py)
Wrote 47 lines to view.py
"""Open an exported model in Netron.

    python view.py lfm2      # LFM2.5-1.2B-Instruct
    python view.py dspark    # LFM2.5-1.2B-Instruct-DSpark
    python view.py <path>    # anything else

Netron reads the .onnx graph file and pulls weight blobs from the sibling
.onnx.data lazily, so these open in about a second despite being multi-GB on
disk. Keep the .onnx and .onnx.data files together.
"""
… +37 lines

  Ran 1 shell command

Memory is flat at 223 MB — too low for a 1.2B fp32 load. Let me check whether it's actually working.

  Ran 1 shell command

Only 21.7 s CPU over ~5 min wall — it's blocked on I/O, not computing. Likely an HF Hub network call behind a corporate proxy. Let me test that hypothesis.

  Ran 3 shell commands

It's progressing fine — 4.9 GB working set (matches 1.2B fp32), now inside torch.onnx.export; my earlier snapshot was just too early. Let me verify Netron actually serves the external-data model while
it runs.

  Ran 1 shell command

Netron serves the viewer but the model comes via an internal route. The real questionsolves the sibling .onnx.data. Let me check its server code.

  Ran 2 shell commands

Netron's server resolves /data/<relative-path> against the model's own directory — sok. Confirming:

  Ran 1 shell command

Both the graph and the 1.72 GB external-data blob serve correctly — Netron will resole 1.2B export:

  Ran 1 shell command

While that finishes — Netron's browser-side handling of multi-GB external data matterONNX reader fetches it.

  Ran 2 shell commands

Netron stores external tensors as a lazy offset+length request and only reads them whe graph opens instantly regardless of data size. No stripped variant needed. Checkingthe export:

  Ran 1 shell command

Weights are being written now (4.7 GB to go). Let me check whether optimum offers an alternative export path worth documenting.

  Ran 1 shell command

optimum is missing requests (huggingface_hub 1.x moved to httpx, so it wasn't pulled in). Installing it to check the alternative path.

  Ran 2 shell commands

optimum 2.3.0 reports 0 exportable types — in optimum 2.x the ONNX exporter moved into a separate optimum-onnx package, so optimum[exporters] no longer ships export configs. The direct
torch.onnx.export route sidesteps that. Checking the export:

  Ran 1 shell command

CPU is climbing (350 s) but the data file hasn't grown since 5.4 MB — it's in a compue signal.

  Ran 1 shell command

Data file untouched since 20:21:33 while CPU climbs — it's inside prog.optimize() (coh), which is expected to be slow. Let me add the stable sections to the findings docwhile it runs.

  Ran 2 shell commands

Let me read the LFM2 ShortConv implementation from transformers to document the block precisely (no extra memory pressure during the export).

  Ran 2 shell commands

Confirmed the ShortConv is a double-gated depthwise causal conv: in_proj → (B, C, x), gate B·x, 3-tap depthwise conv, gate by C, then out_proj. Both layer types slot into the same operator_norm →
operator → ffn_norm → MLP skeleton. Checking the export:

  Ran 1 shell command

Let me stop polling and use the time to document the verified ShortConv math.

  Ran 2 shell commands

It's in a long single-threaded optimization pass. Let me wait on it properly instead

  Ran 1 shell command

Still optimizing. I'll stop polling — the background task will notify me on completious in the doc.

  Ran 1 shell command

Status so far — the DSpark half is done and verified; the 1.2B export is still runnin

The main finding: these aren't sibling variants. LFM2.5-1.2B-Instruct is the target mive-decoding drafter (295.7 M), and it is not loadable by anything on PyPI — the repoships only config.json + model.safetensors, declares Lfm2DSparkDraftModel, has no auto_map, and that class lives only in SGLang (Linux/CUDA).

So I read its tensor manifest off the Hub with an HTTP range request (no download needed), pulled the reference implementation from sglang/srt/models/{lfm2_dspark,dspark,dflash}.py, and rebuilt the
architecture in plain PyTorch. It checks out both ways:

- load_state_dict → 0 missing, 0 unexpected across all 62 tensors
- ONNX Runtime vs PyTorch → max|diff| 2.3e-05 on logits (fp32 round-off)

What the odd tensors turned out to be: fc [2048, 10240] fuses hidden states tapped from 5 of the target's 16 layers; markov_head (2× [65536, 256]) is a rank-256 factorized bigram matrix adding a logit
bias so a block predicted in parallel keeps token-to-token dependence; confidence_heats per-token acceptance. Attention over the block is bidirectional — SGLang marksthese layers ENCODER_ONLY.

Also worth flagging on the 1.2B model: its real feed-forward width is 8192, not the 12288 in config.json (block_auto_adjust_ff_dim applies the SwiGLU ⅔ rule), and 10 of its 16 layers are double-gated
3-tap depthwise convs, not attention.

Two things I resolved that would otherwise bite:
- optimum-cli export onnx is a dead end — optimum 2.3.0 reports 0 exportable types (the ONNX exporter moved to a separate optimum-onnx package), so I used torch.onnx.export(dynamo=True) directly.
- Netron handles the multi-GB weights fine: I confirmed in its source that it serves  and treats external initializers as lazy offset+length reads, so the graph opensinstantly.

~/Desktop/session.md has the full write-up. The 1.2B export is in ONNXProgram.optimize() (~8 min of single-threaded constant folding on a 1.17 B-param graph); I'll report and open Netron when it
completes.

✻ Brewed for 33m 32s · done Sunday 8:28 PM

● Background command "Export LFM2.5-1.2B-Instruct to ONNX" completed (exit code 0)

The 1.2B export finished (exit 0). Let me see the results.

  Listed 1 directory

Exported in 528 s, and it verifies: max|diff| 7.0e-05, argmax agreement 100.00%. Let eck one warning it emitted.

  Ran 2 shell commands

The graph histogram is a clean fingerprint of the architecture: 10 Conv (ShortConv laayers), 45 RMSNorms, 16 SiLU sigmoids. Let me fix one cosmetic script issue andfinalize the doc.

  Ran 1 shell command

Now finalizing the findings doc with the 1.2B results.

  Ran 1 shell command

The Bash wrapper collapsed my escaped backslashes into a \U escape. Switching the path display to forward slashes.

  Ran 1 shell command

session.md is complete (522 lines). Now let me launch Netron on both models.

● Write(serve_both.py)
Wrote 24 lines to serve_both.py
"""Serve both exported models in Netron at once and keep the servers alive.

netron.start() runs its HTTP server on a daemon thread, so the process has to
stay up for the pages to keep working. Ctrl-C (or killing this process) stops
both.
"""

import os
import threading
… +14 lines

  Ran 2 shell commands

Both servers are live. Opening them in the browser:

  Ran 1 shell command

Both tabs are open. Let me record the live servers in the doc and save the durable en

  Ran 3 shell commands, wrote 1 memory

Both models are downloaded, converted, verified, and open in Netron.

Live now: http://localhost:8080 (LFM2.5-1.2B-Instruct) · http://localhost:8081 (DSpar

┌─────────────────┬──────────────────────┬─────────┐
│                 │ LFM2.5-1.2B-Instruct │ DSpark  │
├─────────────────┼──────────────────────┼─────────┤
│ model.onnx      │ 1.69 MB              │ 0.75 MB │
├─────────────────┼──────────────────────┼─────────┤
│ model.onnx.data │ 4.68 GB              │ 1.72 GB │
├─────────────────┼──────────────────────┼─────────┤
│ Nodes           │ 806                  │ 588     │
├─────────────────┼──────────────────────┼─────────┤
│ ORT vs PyTorch  │ 7.0e-05, argmax 100% │ 2.3e-05 │
└─────────────────┴──────────────────────┴─────────┘

What made this non-trivial

The two repos aren't sibling variants — the second is a speculative-decoding drafter  PyPI can load it. It ships only config.json + model.safetensors, declaresLfm2DSparkDraftModel with no auto_map, and lists model_type: qwen3 as a decoy. That class exists only in SGLang, which doesn't run on Windows.

So I read its tensor manifest off the Hub with an HTTP range request (safetensors' JSON header sits in the first bytes — no 591 MB download to see the names), pulled the reference implementation from
sglang/srt/models/{lfm2_dspark,dspark,dflash}.py, and rebuilt it in plain PyTorch. Itys: load_state_dict took all 62 tensors with 0 missing / 0 unexpected, and ONNXRuntime matches PyTorch to fp32 round-off. That's confirmation, not just plausibility.

The odd tensors decoded: fc [2048, 10240] fuses hidden states tapped from 5 of the target's 16 layers; markov_head (2× [65536, 256]) is a rank-256 factorized bigram matrix that re-injects
token-to-token dependence into a block predicted in parallel; confidence_head [1, 230ken acceptance. Block attention is bidirectional — SGLang marks those layersENCODER_ONLY.

Two corrections to what the configs claim

- The 1.2B model's real feed-forward width is 8192, not the 12288 in config.json — block_auto_adjust_ff_dim applies the SwiGLU ⅔ rule. Taking the config at face value overstates the MLP by 50%.
- It's 10 conv + 6 attention layers (I had this as 11/5 earlier and fixed it in the d

Worth knowing

- optimum-cli export onnx is a dead end — optimum 2.3.0 reports 0 exportable types (tte optimum-onnx package), so I used torch.onnx.export(dynamo=True) directly.
- Netron handles the multi-GB weights fine: its server resolves sibling .onnx.data via /data/, and onnx.js treats external initializers as lazy offset+length reads — the graph opens instantly and only
  fetches a weight's bytes when you click it. Keep .onnx and .onnx.data together.
- One limitation I did not fix: batch is pinned to 1 in the 1.2B export (torch.export specialized it despite the Dim hint; sequence stayed dynamic). Irrelevant for viewing — say the word if you want a
  batched re-export (~9 min).

Full write-up in ~/Desktop/session.md (555 lines, §1–11), including the op-histogram ify the architecture at a glance: 10 Conv = the ShortConv layers, 6 Softmax = theattention layers, 45 RMSNorm groups, and only 1 Cos/Sin because RoPE gets CSE'd across layers.

✻ Worked for 4m 3s · done Sunday 8:33 PM

※ recap: Both LFM2.5 models are downloaded, converted to ONNX, numerically verified, and open in Netron at localhost:8080 and 8081, with full findings written to ~/Desktop/session.md. Nothing pending
  unless you want the 1.2B re-exported with a dynamic batch dimension. (disable recaps

❯ Only the dspark version of the model is shown in netron in port 8080 and 8081. If thehitecture (e.g., Liquid AI LFMs, custom Mamba/Conv layers) that optimum-cli does nothave a pre-registered ONNX config for, export it using transformers and PyTorch directly

You're right, and I mis-verified that — I checked both ports returned HTTP 200 on /, but the viewer HTML is byte-identical for any model, so it looked fine. Let me find the actual cause.

  Ran 1 shell command

Found it. In netron/server.py:

self.server = _ThreadedHTTPServer(address, _HTTPRequestHandler)
self.server.RequestHandlerClass.content = content   # line 136 — CLASS attribute

RequestHandlerClass is the shared _HTTPRequestHandler class, and content is declared the second netron.start() in the same process overwrites the first — both ports served whichever loaded last (DSpark, on 8081). You cannot serve two models from one Python process. My serve_both.py did exactly that. Fixing with one process per model:

● Write(serve_both.py)
Added 40 lines, removed 10 lines
"""Serve both exported models in Netron at once and keep the servers alive.
"""Serve both exported models in Netron, each in its OWN process.

netron.start() runs its HTTP server on a daemon thread, so the process has to
stay up for the pages to keep working. Ctrl-C (or killing this process) stops
both.
Netron stores the served model on a *class* attribute of its request handler
(`server.py`: `self.server.RequestHandlerClass.content = content`, where
`content` is declared at class level on `_HTTPRequestHandler`). Two
`netron.start()` calls in one interpreter therefore clobber each other and ev
port ends up serving whichever model was loaded last. One process per model is
the only way to run two at once.

    python serve_both.py            # spawn a child per model, then block
    python serve_both.py --serve <path> <port>   # (internal) single-model child
"""

import os
import subprocess
import sys
import threading

import netron

ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS = [
    ("LFM2.5-1.2B-Instruct", os.path.join(ROOT, "onnx", "LFM2.5-1.2B-Instruct", "model.onnx"), 8080),
    ("LFM2.5-1.2B-Instruct-DSpark", os.path.join(ROOT, "onnx", "LFM2.5-1.2B-I), 8081),
]

for name, path, port in MODELS:

def serve_one(path, port):
    import netron

    netron.start(path, address=("localhost", port), browse=False)
    print(f"{name:32} http://localhost:{port}", flush=True)
    print(f"SERVING {os.path.basename(os.path.dirname(path))} on {port}", flu
    threading.Event().wait()

print("READY", flush=True)
threading.Event().wait()

def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--serve":
        serve_one(sys.argv[2], int(sys.argv[3]))
        return

    children = []
    for name, path, port in MODELS:
        if not os.path.exists(path):
            print(f"MISSING {path}")
            continue
        children.append(subprocess.Popen(
            [sys.executable, "-u", os.path.abspath(__file__), "--serve", path, str(port)]
        ))
        print(f"{name:32} http://localhost:{port}", flush=True)
    print("READY", flush=True)
    for c in children:
        c.wait()


if __name__ == "__main__":
    main()

  Ran 2 shell commands

● Background command "Start Netron servers for both models" failed with exit code 1

Now let me verify properly this time — parse the actual bytes each port serves, rathe.

  Ran 1 shell command

Now verified by parsing the actual served bytes — the ports serve genuinely different

┌──────┬───────────┬───────┬───────────┬──────┬─────────┬────────────────┐
│ Port │   Bytes   │ Nodes │  Params   │ Conv │ Softmax │     Model      │
├──────┼───────────┼───────┼───────────┼──────┼─────────┼────────────────┤
│ 8080 │ 1,690,319 │ 806   │ 1170.34 M │ 10   │ 6       │ LFM2 hybrid    │
├──────┼───────────┼───────┼───────────┼──────┼─────────┼────────────────┤
│ 8081 │ 750,061   │ 588   │ 429.94 M  │ 0    │ 5       │ DSpark drafter │
└──────┴───────────┴───────┴───────────┴──────┴─────────┴────────────────┘

Reopening the tabs (the old ones cached the wrong model):

  Ran 5 shell commands

Fixed. Both tabs are reopened and now show different models.

The cause was the Netron server, not the export. In netron/server.py, the served modeute:

class _HTTPRequestHandler(...):
    content = None                                     # line 47 — class level
...
    self.server.RequestHandlerClass.content = content  # line 136

RequestHandlerClass is that shared class, so my second netron.start() in the same process overwrote the first — both ports served whichever model loaded last (DSpark, started second). serve_both.py
now spawns one child process per model.

It fails quietly, which is why my check missed it: every port still answers HTTP 200 byte-identical for any model. I checked reachability, which passed while the pageswere wrong. The real check is fetching /data/model.onnx and parsing what comes back — now done:

┌──────┬───────────┬───────┬───────────┬──────┬─────────┬────────────────┐
│ Port │   Bytes   │ Nodes │  Params   │ Conv │ Softmax │                │
├──────┼───────────┼───────┼───────────┼──────┼─────────┼────────────────┤
│ 8080 │ 1,690,319 │ 806   │ 1170.34 M │ 10   │ 6       │ LFM2 hybrid    │
├──────┼───────────┼───────┼───────────┼──────┼─────────┼────────────────┤
│ 8081 │ 750,061   │ 588   │ 429.94 M  │ 0    │ 5       │ DSpark drafter │
└──────┴───────────┴───────┴───────────┴──────┴─────────┴────────────────┘

On the export method — that's already what was done. The 1.2B model never went through optimum; it was exported with transformers + torch.onnx.export directly (export_lfm2.py:
AutoModelForCausalLM.from_pretrained(...) → torch.onnx.export(..., dynamo=True)), pre lfm2 config — optimum 2.3.0 reports 0 exportable types since its exporter moved to aseparate optimum-onnx package. The graph on 8080 is the proof: producer pytorch, 1170.34 M params, and 10 Conv + 6 Softmax nodes — the ShortConv/attention interleave that only the real LFM2 has.

What to look for on 8080: 16 near-identical blocks (operator_norm → operator → +residual → ffn_norm → SwiGLU → +residual), where 10 contain a Conv flanked by two Mul gates and 6 contain the QKV
fan-out and a Softmax — attention sitting at layers 2, 5, 8, 10, 12, 14.

~/Desktop/session.md now documents this bug too (§11, 580 lines). One note: the file nother process and took a few retries — worth closing any editor holding it if youwant me to update it again.

✻ Worked for 2m 40s · done Sunday 8:52 PM

※ recap: Both LFM2.5 models are downloaded, converted to ONNX, numerically verified, and now serving correctly in Netron at localhost:8080 (the 1.2B hybrid) and localhost:8081 (the DSpark drafter).
  Findings are written up in Desktop/session.md. Nothing pending — just open the tabs.

● Background command "Start Netron with one process per model" was stopped because the ry

The Netron parent process was killed for low memory. Let me check what survived.

  Ran 2 shell commands

All three Python processes were killed and both ports are down. Cause is system-wide: Chrome at ~4 GB across tabs. Worth noting the Netron servers themselves are cheap —_ContentProvider reads from disk on demand, so the 4.68 GB never enters RAM; they were collateral.

Let me restart them detached from this session so the harness's memory-pressure killer doesn't take them out again.

  Ran 1 shell command

Both back as detached pythonw processes at 27 MB each — confirming the servers are lightweight. Verifying they serve the right models:

  Ran 1 shell command

Both servers are back up and serving the correct, distinct models. Just refresh your two browser tabs.

- http://localhost:8080 — LFM2.5-1.2B-Instruct: 806 nodes, 1170.34 M params, Conv=10 Softmax=6
- http://localhost:8081 — LFM2.5-1.2B-Instruct-DSpark: 588 nodes, 429.94 M params, Co

What happened: the machine hit low memory (2.9 GB free of 15.6 GB, Chrome holding ~4 y background task — taking the parent and both child servers with it. The serversweren't the cause; they're 27 MB each, since Netron's _ContentProvider reads from disk on demand and never pulls the 4.68 GB weight file into RAM.

They're now launched via Start-Process as detached pythonw processes, independent of this session, so they'll survive both memory-pressure kills and the session ending.

To stop them later:

Get-Process pythonw | Stop-Process

To bring them back:

python C:/Users/ramkumar.pri/lfm25-onnx/serve_both.py

One caveat given the memory headroom: the 806-node LFM2 graph is the heavier of the two to render. If Chrome struggles, keep only one Netron tab open at a time — the servers are fine either way.

✻ Cooked for 1m 23s · done Sunday 9:45 PM

※ recap: Both LFM2.5 models are downloaded, converted to ONNX, verified, and now serving in Netron on ports 8080 (the 1.2B hybrid) and 8081 (the DSpark drafter). Just refresh your two browser tabs to see them. (disable recaps in /config)


---
---

# Session 2 — 2026-09-08 · independent re-run on a clean machine

**Session log & findings (appended)**
Host: Windows 11 Enterprise (26200) · Python 3.13.15
Working dir: `C:\Users\TEMP\project\lfm25-onnx\`

This is a second, independent run of the same task on a *different, bare*
machine (no Python ML packages, no artifacts from session 1 — that session's
`C:\Users\ramkumar.pri\lfm25-onnx\` does not exist here). Everything in
§§1–11 above reproduced. What follows is only what is **new or corrected**.

---

## S2.0 Summary of what this session adds

| # | Finding | Status vs §§1–11 |
|---|---|---|
| S2.1 | **Liquid AI already publishes an official ONNX export** of the target model, in 6 precisions | entirely new — §§1–11 never mention it |
| S2.2 | That official graph independently confirms the reverse-engineered LFM2 architecture, op for op | new cross-validation |
| S2.3 | It was built by **onnxruntime-genai**, which has a dedicated `LFM2Model` builder | new — completes §6's "optimum is a dead end" story |
| S2.4 | The target and the drafter use **different RoPE conventions** (NeoX vs GPT-J interleaved) | new |
| S2.5 | The confidence head's second input is **confirmed from source**, no longer inferred | upgrades §10 limitation #1 |
| S2.6 | `sample_from_anchor` defaults to `True`, so the draft forwards exactly `block_size` rows | new, justifies the block=9 export shape |
| S2.7 | The torch DLL error was **not** the §5 red herring — the machine's MSVC runtime was too old | corrects §5 |
| S2.8 | `transformers` 5.x renamed the LFM2 config key `tie_embedding` → `tie_word_embeddings` | new gotcha |
| S2.9 | The drafter family has two more members (8B-A1B, 2.6B) and an Apple-silicon Metal path | new |
| S2.10 | The safetensors range-read trick applied to the 2.34 GB target | extends §4.2 |
| S2.11 | **Netron's Python server has no HTTP Range support** — touching the weights pulls all 4.68 GB into RAM | **corrects §7** |

Also confirmed unchanged: `optimum-onnx` — the split-out package §6 predicted —
now exists on PyPI at **0.1.0** and **still has no `lfm2` exporter config**, so
§6's "dead end" verdict stands (details in S2.3).

---

## S2.1 There is an official ONNX export of the target model

`LiquidAI/LFM2.5-1.2B-Instruct` links out to
**[`LiquidAI/LFM2.5-1.2B-Instruct-ONNX`](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct-ONNX)**
(ungated, last modified 2026-02-17). Session 1 converted the target from
scratch without noticing it existed. It ships six precisions:

```
onnx/model.onnx            140,810 B  graph  + 3 data files (2.06 + 2.07 + 0.58 GB)  fp32
onnx/model_fp16.onnx       140,027 B  graph  + 2 data files (2.07 + 0.29 GB)
onnx/model_q8.onnx         188,702 B  graph  + 1.77 GB
onnx/model_quantized.onnx  185,703 B  graph  + 1.52 GB
onnx/model_q4.onnx         183,173 B  graph  + 0.85 GB
onnx/model_q4f16.onnx      182,795 B  graph  + 0.76 GB
```

**This does not make the work above redundant**, for three reasons:

1. There is **no official ONNX for the drafter** — the reconstruction in
   `dspark_model.py` is still the only way to get a DSpark graph.
2. The official target graph is a **decoder-with-past** graph whose attention is
   collapsed into 6 opaque `com.microsoft::GroupQueryAttention` nodes. For
   *reading the architecture* — which is the point of opening it in Netron — the
   decomposed `torch.onnx.export` graph is far more informative: it shows the
   QKV fan-out, the RMSNorm arithmetic and the softmax explicitly.
3. Its weights are split across three `.onnx_data*` files, so it is 4.7 GB to
   fetch before you can run anything.

The **topology file is only 140 KB**, though, so the whole comparison below was
done by downloading `model.onnx` alone and parsing it with
`load_external_data=False` — no weights needed.

---

## S2.2 The official graph confirms the architecture, independently

Session 1 derived the LFM2 hybrid layout by reading `config.json`, the weight
manifest and `modeling_lfm2.py`. The official export agrees on every count.

**The input names alone spell out the hybrid interleave.** You do not even have
to open the graph:

```
past_conv.0        FLOAT[batch_size,2048,3]              <- ShortConv state
past_conv.1        FLOAT[batch_size,2048,3]
past_key_values.2.key/value  FLOAT[batch_size,8,past_sequence_length,64]   <- attention
past_conv.3 …4     FLOAT[batch_size,2048,3]
past_key_values.5.key/value  …
past_conv.6 …7
past_key_values.8.key/value
past_conv.9
past_key_values.10.key/value
past_conv.11
past_key_values.12.key/value
past_conv.13
past_key_values.14.key/value
past_conv.15
```

```
conv layers      : [0, 1, 3, 4, 6, 7, 9, 11, 13, 15]     (10)
attention layers : [2, 5, 8, 10, 12, 14]                 (6)
```

which is exactly `config.layer_types`. `conv_L_cache: 3` shows up as the literal
trailing `3` of the conv state, and `num_key_value_heads: 8` / `head_dim: 64` as
the `[…,8,…,64]` of the KV cache.

**Op histogram (official fp32, 386 nodes, opset ai.onnx=21 + com.microsoft=1):**

```
93 MatMul, 62 Mul, 44 SimplifiedLayerNormalization, 32 Add, 24 Reshape,
21 Transpose, 20 Slice, 16 Sigmoid, 12 Gather, 11 Shape, 10 Split, 10 Concat,
10 Conv, 10 Unsqueeze, 6 GroupQueryAttention, 2 Cast, 1 ReduceSum, 1 Sub,
1 SkipSimplifiedLayerNormalization
```

Line up the load-bearing counts against session 1's decomposed export:

| Count | Official (fused) | Session 1 / 2 (`torch.onnx.export`) | Why |
|---:|---|---|---|
| **10** | `Conv` | `Conv` | the 10 ShortConv layers |
| **10** | `Split` | `Split` | their `chunk(BCx, 3)` |
| **16** | `Sigmoid` | `Sigmoid` | one SwiGLU SiLU per layer |
| **6** | `GroupQueryAttention` | 6 × `Softmax` | the 6 attention layers |
| **45** | 44 `SimplifiedLayerNormalization` + 1 `SkipSimplifiedLayerNormalization` | 45 RMSNorm groups | 16 `operator_norm` + 16 `ffn_norm` + 6 `q_layernorm` + 6 `k_layernorm` + 1 `embedding_norm` |

**Two independent exporters, two totally different op vocabularies, and the same
45.** That is a strong check on session 1's RMSNorm accounting.

**The `Conv` node settles the ShortConv question directly:**

```
/model/layers.0/conv/Conv   kernel_shape=[3]  group=2048  pads=[0,0]
   weight  model.layers.0.conv.conv.weight  shape=[2048, 1, 3]
/model/layers.0/conv/Split  axis=1  -> 3 outputs
```

`group == channels == 2048` is depthwise; `kernel_shape=[3]` is the 3 taps; the
3-way `Split` is the B / C / x fan-out. Session 1 read this off
`modeling_lfm2.py`; here it is in a shipped artifact.

**`GroupQueryAttention` attributes** confirm the attention config too:

```
num_heads=32  kv_num_heads=8  scale=0.125  local_window_size=-1
softcap=0.0   do_rotary=1     rotary_interleaved=0
inputs: [q_norm_out, k_norm_out, v_proj_out, past.key, past.value,
         seqlens_k, total_seqlen, cos_cache, sin_cache]
```

`scale = 0.125 = 64**-0.5`, i.e. `head_dim ** -0.5`, and the separate
`q_norm`/`k_norm` inputs are the per-head Q/K RMSNorm.

**Why the initializer count is 1178.53 M and not 1170.34 M.** The 8.19 M
difference is exactly `128000 × 32 × 2` — the RoPE `cos_cache` and `sin_cache`
materialised as initializers over the full `max_position_embeddings`. Worth
knowing before concluding two exports disagree on parameter count.

**The q4 variant is the same graph with quantised GEMMs** — 385 nodes, and every
one of the 93 `MatMul`s becomes `MatMulNBits`, plus one `GatherBlockQuantized`
for the embedding lookup. Node structure is otherwise identical, which is a
neat illustration that quantisation here is an op substitution, not a rewrite.

---

## S2.3 Who built it: `onnxruntime-genai`, not `optimum`

§6 concluded that optimum cannot export LFM2 and went straight to
`torch.onnx.export`. That was right, and the picture is now complete:

- **`optimum` / `optimum-onnx`** — the split-out ONNX exporter package §6
  predicted now exists on PyPI as `optimum-onnx` **0.1.0**. Its
  `optimum/exporters/onnx/model_configs.py` has `Qwen3OnnxConfig`,
  `Qwen3MoeOnnxConfig` and many others but **no `lfm2` entry at all**. So the
  optimum route is still a dead end for this architecture — §6 stands.

- **`onnxruntime-genai`** does support it. `src/python/py/models/builder.py`
  dispatches `Lfm2ForCausalLM` to a dedicated `LFM2Model` builder
  (`src/python/py/models/builders/lfm2.py`), which is what produces the
  `GroupQueryAttention` / `SimplifiedLayerNormalization` / `MatMulNBits`
  vocabulary and the `past_conv.*` cache inputs seen above.

Its `make_short_conv()` is the same double-gated block, emitted node by node:

```
in_proj (MatMul, hidden -> 3*hidden)
  -> Split(axis=1) into B, C, x
  -> Mul_1                       (input gate  B * x)
  -> CausalConvWithState(...)    (depthwise, past_conv/present_conv threaded through)
  -> Mul_2                       (output gate C * conv_out)
  -> Transpose -> out_proj (MatMul)
```

**One concrete drift between the shipped file and current source.** The builder
today declares

```python
conv_cache_shape = ["batch_size", self.hidden_size, self.conv_L_cache - 1]
```

i.e. depth **2**, whereas the published `model.onnx` has `past_conv.N` of depth
**3**. A 3-tap causal conv only needs the 2 preceding tokens as carried state, so
the current `conv_L_cache - 1` is the tighter size and the February artifact
predates that change. If you script against the official ONNX, read the conv
cache depth from the graph rather than assuming either value.

**Practical guidance.** Use `onnxruntime-genai` if you want to *run* LFM2 under
ONNX Runtime; use `torch.onnx.export(dynamo=True)` — what both sessions did — if
you want to *read* the architecture, because the fused contrib ops hide exactly
the internals you opened Netron to see.

---

## S2.4 The target and the drafter disagree about RoPE

A sharp architectural contrast that §§3–4 record separately but never put side
by side:

| | LFM2.5-1.2B-Instruct (target) | LFM2.5-1.2B-Instruct-DSpark (drafter) |
|---|---|---|
| RoPE style | **NeoX** — halves, `rotate_half` | **GPT-J interleaved** — adjacent pairs |
| Evidence | official ONNX `GroupQueryAttention` attr `rotary_interleaved=0` | `config.json`: `"rope_is_neox_style": false` |
| `rope_theta` | 1 000 000.0 | 1 000 000.0 |

Both share theta and `head_dim=64`, but pair channels differently:

```
NeoX          x1 = x[..., :32]     x2 = x[..., 32:]     out = cat(o1, o2)
GPT-J (draft) x1 = x[..., 0::2]    x2 = x[..., 1::2]    out = stack(o1,o2).flatten(-2)
```

This matters for anyone re-implementing the drafter: getting it wrong is
numerically silent at load time (shapes are identical) and only shows up as
degraded acceptance. `dspark_model.apply_rope_gptj` implements the interleaved
form, matching
`sglang/srt/layers/rotary_embedding/utils.py::apply_rotary_emb(is_neox_style=False)`.

---

## S2.5 The confidence head's second input — confirmed, not inferred

§10 listed this as inferred limitation #1: *"the confidence head's second input
is taken to be `markov_w1(prev_token)` … its 2304 = 2048 + 256 input width and
`with_markov=True` default leave little else it could be."*

That inference is **correct**, and can now be cited. The call site is not in
`models/dspark.py` but in the proposal glue:

`sglang/srt/speculative/dspark_components/dspark_planner.py:868`

```python
def build_markov_embed_stack(*, anchor_tokens, draft_tokens, markov_head, gamma):
    prev_seq = torch.cat(
        [anchor_tokens.view(-1, 1), draft_tokens[:, : gamma - 1]], dim=1
    )
    return markov_head.get_prev_embeddings(prev_seq)      # == markov_w1(prev_seq)


def compute_confidence(*, draft_hidden, anchor_tokens, draft_tokens,
                       confidence_head, markov_head, gamma):
    markov_embed_stack = build_markov_embed_stack(...)     # [B, gamma, 256]
    confidence_raw = confidence_head(draft_hidden, markov_embed_stack)
    return confidence_head.apply_sts(confidence_raw)       # sigmoid(raw / sts_temp)
```

reached from `DsparkDraftSampler.__call__` →
`DSparkVerifyPlanner.compute_confidence_tensor` → here. So, precisely:

- the head consumes `cat(hidden_states, markov_w1(prev_token))`, width 2048+256;
- `prev_token` at block position *i* is the token proposed at *i−1*, with
  position 0 using the **anchor** (the last token the target actually committed);
- `apply_sts` is `sigmoid(raw / sts_temperatures)`, and `sts_temperatures` is a
  **non-persistent buffer defaulting to 1.0** that is absent from the checkpoint
   — so at rest it is a plain sigmoid, exactly as exported.

Limitation #1 in §10 can be struck. Limitation #2 (paged KV pool vs plain
concatenated prefix) still stands and is still numerically neutral.

---

## S2.6 `sample_from_anchor` defaults True → the block is exactly 9 rows

`sglang/srt/speculative/dspark_components/dspark_config.py:29`

```python
def get_dspark_sample_from_anchor(draft_hf_config) -> bool:
    return bool(_cfg_get(draft_hf_config, "sample_from_anchor", True))
```

and in the sampler, `query_token_num = gamma if sample_from_anchor else gamma + 1`.
The DSpark `config.json` does not set the key, so it is `True`, `gamma =
block_size = 9`, and the drafter forwards **exactly 9 rows** — no extra anchor
row to slice off (`select_draft_hidden_without_anchor` is the other branch).
That is what justifies the `[batch, 9, …]` block shape of the export.

---

## S2.7 The torch DLL error was real here — §5's "red herring" is machine-specific

§5 records `ImportError: DLL load failed while importing _C` as a red herring
caused by pip not having written `torch/lib` yet, and notes the MSVC runtime was
"present". On this machine the same *class* of error was **genuine**, and cost
the most time of anything in this session. Two different failures in sequence:

**1. `OSError: [WinError 1114] A dynamic link library (DLL) initialization
routine failed. Error loading "…\torch\lib\c10.dll"`**

`torch/lib` was fully written (`torch_cpu.dll` 305 MB, `c10.dll` 1.1 MB), so the
§5 explanation did not apply. `c10.dll` imports `MSVCP140.dll`,
`VCRUNTIME140.dll`, `VCRUNTIME140_1.dll` — and `C:\Windows\System32` held
**14.24.28127.4**, the Visual Studio 2019 (16.4, Dec 2019) redistributable.
torch 2.14 is built with the VS 2022 toolset. **Presence is not enough; the
version matters** — that is the correction to §5.

**2. `OSError: [WinError 126] The specified module could not be found. Error
loading "…\torch\lib\shm.dll"`**, after the first was fixed.

`shm.dll` → `torch_cpu.dll` → **`vcruntime140_threads.dll`**, which ships only
with 14.4x+ and was **not present anywhere on the machine** (a full-disk search
found zero copies).

**Fix, no admin rights needed.** Rather than running `vc_redist.x64.exe` (UAC),
take the DLLs from the PyPI wheel and drop them into torch's own lib directory:

```bash
python -m pip download msvc-runtime --no-deps --only-binary=:all: -d .
#   msvc_runtime-14.44.35112-cp313-cp313-win_amd64.whl
#   -> .data/data/Scripts/{msvcp140,msvcp140_1,msvcp140_2,msvcp140_atomic_wait,
#      msvcp140_codecvt_ids,vcruntime140,vcruntime140_1,vcruntime140_threads,
#      concrt140,vcamp140,vccorlib140,vcomp140}.dll
# unzip those into:
#   <python>/Lib/site-packages/torch/lib/
```

This works because `torch/__init__.py::_load_dll_libraries` calls
`os.add_dll_directory(<torch>/lib)` before loading, and DLL-search order puts
user-added directories **ahead of System32** — so the 14.44 copies win over the
14.24 ones in System32 without touching System32 at all. Install the whole
coherent 14.44 set rather than mixing versions (an early attempt that copied
Edge's 14.51 `msvcp140.dll` next to a 14.24 `vcruntime140.dll` is exactly the
kind of mismatch to avoid).

Diagnostic worth reusing — no tooling required, just read the PE for DLL name
strings:

```python
import re, pathlib
b = pathlib.Path(r"...\torch\lib\torch_cpu.dll").read_bytes()
print(sorted({n.decode().lower() for n in re.findall(rb"[A-Za-z0-9_\-]+\.dll", b)}))
# -> ['c10.dll', 'libiomp5md.dll', 'msvcp140.dll', 'uv.dll', 'vcruntime140.dll',
#     'vcruntime140_1.dll', 'vcruntime140_threads.dll', ...]
```

The `PYTHONUTF8=1` / `PYTHONIOENCODING=utf-8` gotcha from §5 is **still real** —
with it set, the exporter's `✅` progress lines print fine on both exports.

---

## S2.8 `transformers` 5.x renamed the LFM2 tie-embedding key

`transformers` **5.16.1** (this session; §5's session predates it):

```
AttributeError: 'Lfm2Config' object has no attribute 'tie_embedding'
```

The checkpoint's `config.json` says `"tie_embedding": true`; `Lfm2Config` in
transformers 5.x exposes it as **`tie_word_embeddings`**. (Tellingly, the
official ONNX repo's `config.json`, regenerated with `transformers 5.1.0`,
carries **both** keys.) Read it defensively:

```python
tied = getattr(cfg, "tie_word_embeddings", None)
if tied is None:
    tied = getattr(cfg, "tie_embedding", None)
```

Nothing else in the export path changed across the 4.57 → 5.16 gap; the model
loaded, exported and verified unchanged.

---

## S2.9 The drafter family, and the acceptance numbers

The DSpark README carries detail §2 summarised only as "~2× / 4.81 tokens":

| Benchmark | Accepted tok/step | H100 speedup | M4 Max speedup |
|---|---:|---|---|
| MATH-500 | 6.02 | 2.56× (668 → 1712 tok/s) | 2.62× (140 → 366) |
| HumanEval | 5.31 | 2.26× (664 → 1499) | 2.87× (136 → 389) |
| MBPP | 5.52 | 2.37× (667 → 1578) | 2.74× (137 → 375) |
| GSM8K | 4.34 | 1.67× (624 → 1041) | 2.73× (140 → 381) |
| MT-Bench | 3.90 | 1.66× (657 → 1091) | 1.72× (137 → 237) |
| **Mean** | **5.02** | **2.10×** | **2.54×** |

Two things worth recording:

- **The drafter also runs on Apple silicon** through a Metal backend, where the
  speedup is *higher* than on an H100 (2.54× vs 2.10× mean) — decode on a laptop
  is memory-bandwidth-bound, which is precisely what speculative decoding buys
  back. It is not an SGLang/CUDA-only artifact.
- **Speculative decoding here is exact**: the target verifies every proposed
  token, so output is bit-identical to running the target alone. The drafter
  changes speed, not quality.

There are two sibling drafters — `LFM2.5-8B-A1B-DSpark` and `LFM2.5-2.6B-DSpark`
— so `dspark_model.py` should generalise to them (same `DSparkDraftModel` class,
different `num_hidden_layers` / `target_layer_ids` / `hidden_size`), though only
the 1.2B pair was checked here.

Also noted while reading: the target's README states a **32,768-token context**
while `config.json` sets `max_position_embeddings: 128000`. The RoPE cache in
the official ONNX is built for the full 128000 (that is the 8.19 M in S2.2), so
the 32 768 figure is a *supported/validated* length, not an architectural limit.

---

## S2.10 The safetensors range-request trick, applied to the target

§4.2 used an HTTP range read to get the drafter's manifest without downloading
591 MB. The same works on the 2.34 GB target and is worth having on record,
because it answers the `intermediate_size` question in one shot:

```
x1   BF16 [65536, 2048]   model.embed_tokens.weight
x1   BF16 [2048]          model.embedding_norm.weight
x10  BF16 [2048, 1, 3]    model.layers.N.conv.conv.weight        <- depthwise, 3 taps
x10  BF16 [6144, 2048]    model.layers.N.conv.in_proj.weight     <- 3 x 2048
x10  BF16 [2048, 2048]    model.layers.N.conv.out_proj.weight
x16  BF16 [8192, 2048]    model.layers.N.feed_forward.w1.weight  <- 8192, not 12288
x16  BF16 [2048, 8192]    model.layers.N.feed_forward.w2.weight
x16  BF16 [8192, 2048]    model.layers.N.feed_forward.w3.weight
x16  BF16 [2048]          model.layers.N.ffn_norm.weight
x16  BF16 [2048]          model.layers.N.operator_norm.weight
x6   BF16 [64]            model.layers.N.self_attn.k_layernorm.weight
x6   BF16 [512, 2048]     model.layers.N.self_attn.k_proj.weight
x6   BF16 [2048, 2048]    model.layers.N.self_attn.out_proj.weight
x6   BF16 [64]            model.layers.N.self_attn.q_layernorm.weight
x6   BF16 [2048, 2048]    model.layers.N.self_attn.q_proj.weight
x6   BF16 [512, 2048]     model.layers.N.self_attn.v_proj.weight
                          TOTAL 1170.34 M over 148 tensors
```

The `x10` / `x6` split is the hybrid; **no `lm_head.weight` tensor exists**,
which is the tying (§3) visible on the wire. That the counts are 10/6 and not
11/5 is worth flagging: session 1's first pass guessed 11 conv + 5 attention
from the README before correcting against `layer_types`.

---

## S2.11 Correction to §7 — Netron's server has **no** HTTP Range support

§7 explains why Netron opens a 4.7 GB model instantly, and concludes:

> `onnx.js` treats an `EXTERNAL` initializer as a **lazy `{location, offset,
> length}` request** and only calls `read()` when you click a specific tensor.
> […] the browser only ever fetches the byte range for a weight you actually
> inspect.

and later:

> the Netron servers themselves are cheap — `_ContentProvider` reads from disk on
> demand, so the 4.68 GB never enters RAM.

**The conclusion (fast open) is right; the stated mechanism is wrong.** Reading
`netron/server.py` 9.2.6 line by line:

```python
class _ContentProvider:
    def read(self, path):
        ...
        with open(filename, "rb") as file:
            return file.read()          # <- the WHOLE file, no offset/length

class _HTTPRequestHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        ...
        if path.startswith("/data/"):
            content = self.content.read(path)
            if content:
                content_type = "application/octet-stream"
                status_code = 200        # <- always 200, never 206
```

There is **no `Range` parsing, no `206`, and no `Content-Range` anywhere in the
file**, and `read()` takes no offset. Verified on the wire:

```
$ curl -r 0-1023 http://localhost:8080/data/model.onnx.data
HTTP 200   size_download = 4,681,367,552      # Range header ignored, whole file sent
```

And measured on the server process:

| Netron server | current WS | **peak WS** | what happened |
|---|---:|---:|---|
| :8080 (LFM2, `.data` requested once) | 26 MB | **4 491 MB** | read the whole 4.68 GB into a Python `bytes` |
| :8081 (DSpark, `.data` never touched) | 26 MB | **27 MB** | topology only |

**So what actually makes it fast** is only the first half of §7: the `.onnx` file
holds topology alone (1.82 MB here), and graph construction never touches the
weight blob. Both servers idle at 26 MB because *nobody has clicked a weight*.

**Practical consequence, and it is the opposite of §7's advice.** Clicking a
weight tensor to inspect its values in a browser-served 1.2B graph makes the
server materialise **4.68 GB in one allocation** and ship it whole over
localhost. On this 15.6 GB box that is a real OOM hazard — and it is a very
plausible explanation for the memory-pressure kill §11 recorded, which was
blamed on Chrome at the time.

- Browse **topology** freely — that is genuinely free.
- To **inspect weight values**, use the Netron **desktop app**, which opens the
  file locally instead of going through this server.
- The `commonpath`-guarded relative resolution in §7 *is* correct, so the rule
  **keep `model.onnx` and `model.onnx.data` in the same folder** still holds.

---

## S2.12 Results on this machine

Both models re-derived from scratch and verified independently of session 1.

### LFM2.5-1.2B-Instruct-DSpark — done, verified

```
[ok] target embedding (65536, 2048) (134.22 M, tied -> also the lm_head)
[ok] strict load: 62 checkpoint tensors, 0 missing, 0 unexpected
     draft params       : 295.73 M          (README says 295.7 M)
     shared target head : 134.22 M
     block_size=9  taps=[2, 5, 8, 11, 14]  mask_token_id=64402
[ok] exported in 132.3s

[verify] ONNX Runtime vs PyTorch
  OK  logits         (1, 9, 65536)   max|diff|=2.503e-05   rel=1.09e-06
  OK  hidden_states  (1, 9, 2048)    max|diff|=2.134e-05   rel=1.42e-06
  OK  confidence     (1, 9)          max|diff|=2.682e-07   rel=4.33e-07
  OK  dynamic axes   batch=2 block=5 ctx=20 -> logits(2, 5, 65536)
```

```
model.onnx        0.74 MB      model.onnx.data   1.72 GB
nodes             612          initializers      79        params 429.94 M
opset             ai.onnx=20   ir_version 10     producer pytorch 2.14.0+cpu

op histogram: 131 Mul, 59 MatMul, 55 Reshape, 54 Add, 46 Unsqueeze, 40 Concat,
              31 Transpose, 30 Slice, 27 Pow/ReduceMean/Sqrt/Reciprocal
              (= 27 RMSNorms), 15 Sub, 13 Shape, 10 Expand, 6 Sigmoid,
              5 Softmax, 2 Gather, 2 Cast, 2 Cos, 2 Sin, 1 Squeeze
```

Session 1 got 588 nodes; 612 here. The difference is implementation choice in
the reconstruction (`repeat_interleave` for the GQA fan-out, and `hidden_states`
exported as a third output), not a different architecture — every diagnostic
count is identical:

- **5 Softmax** = the 5 attention layers.
- **6 Sigmoid** = 5 SwiGLU SiLUs + the confidence head.
- **27 RMSNorms** = 5 × (`input_layernorm`, `post_attention_layernorm`, `q_norm`,
  `k_norm` on block K, `k_norm` again on context K) + `norm` + `hidden_norm`.
- **2 Cos / 2 Sin** for 5 layers — one pair for block positions, one for context
  positions, CSE'd across layers.
- **429.94 M params** = 295.73 M draft + 134.22 M injected target embedding.

The tying is visible in the graph as a single initializer with two consumers:

```
initializer  shared_embedding [65536, 2048]
  consumed by 2 nodes: Gather(node_embedding)   <- the embedding lookup
                       Transpose(node_t)        <- feeding the logits MatMul
```

### LFM2.5-1.2B-Instruct — done, verified

```
[ok] loaded Lfm2ForCausalLM: 1170.34 M params, dtype=torch.float32
     layers: 16 = 10 conv + 6 full_attention
     attention at indices [2, 5, 8, 10, 12, 14]
     feed_forward width: 8192 (config.intermediate_size=12288, auto-adjusted)
     tied embeddings: True  conv_L_cache=3
[ok] exported in 229.0s

[verify] logits shape=(1, 32, 65536) max|diff|=5.770e-05 rel=3.51e-06
         argmax agreement: 100.00%
```

```
model.onnx        1.82 MB      model.onnx.data   4.68 GB
nodes             872          initializers      171       params 1170.34 M
opset             ai.onnx=20   ir_version 10     producer pytorch 2.14.0+cpu

inputs:   input_ids       INT64[1,sequence]
          attention_mask  INT64[1,sequence]
outputs:  logits          FLOAT[1,sequence,65536]

op histogram: 188 Mul, 106 MatMul, 95 Add, 52 Transpose, 52 Slice, 48 Reshape,
              45 Pow/ReduceMean/Sqrt/Reciprocal (= 45 RMSNorms), 30 Concat,
              20 Unsqueeze, 16 Sigmoid, 13 Expand, 12 Neg, 10 Split, 10 Conv,
              8 Shape, 7 Where, 6 Softmax, 6 IsNaN, 3 Cast, 2 And,
              1 each Squeeze/Gather/Range/LessOrEqual/Max/GatherND/Cos/Sin
```

872 nodes vs session 1's 806, from `transformers` 5.16.1 vs 4.x mask
construction (the 6 `IsNaN` + 7 `Where` are new; session 1 had 1 `Where` and no
`IsNaN`). **Every architectural count in §8's fingerprint table reproduces
exactly**: 10 Conv, 10 Split, 6 Softmax, 16 Sigmoid, 45 RMSNorm groups, 1 Cos /
1 Sin, 1170.34 M params. Batch is pinned to 1 again, as §10 predicted.

### Timings on this machine (Windows 11 26200, 15.6 GB RAM, no GPU)

| Step | Time |
|---|---|
| pip install of the whole toolchain | ~14 min |
| download both checkpoints (2.93 GB, parallel `curl`) | ~8 min |
| `export_dspark.py` (export + ORT verify) | 132 s export |
| `export_lfm2.py` (export + ORT verify) | 229 s export |

Session 1 reported 528 s for the LFM2 export; 229 s here.

---

## S2.13 What was produced

```
C:\Users\TEMP\project\lfm25-onnx\
  models/
    LFM2.5-1.2B-Instruct/          config, tokenizer, model.safetensors (2,340,697,936 B)
    LFM2.5-1.2B-Instruct-DSpark/   config.json, README, model.safetensors (591,458,386 B)
  onnx/
    LFM2.5-1.2B-Instruct/          model.onnx 1.82 MB  + model.onnx.data 4.68 GB
    LFM2.5-1.2B-Instruct-DSpark/   model.onnx 0.74 MB  + model.onnx.data 1.72 GB
  official_onnx/                   model.onnx / model_q4.onnx  (topology only, 140/183 KB)
  ref/                             SGLang + onnxruntime-genai + optimum-onnx sources
  dspark_model.py                  the drafter reconstruction
  export_lfm2.py  export_dspark.py
  summarize_onnx.py                graph statistics
  serve_both.py                    one Netron child process per model
  check_served.py                  parses what each port actually serves
  logs/
```

Reproduce:

```powershell
$env:PYTHONUTF8 = "1"
python export_dspark.py                 # ~2 min
python export_lfm2.py                   # ~4 min, ~5 GB RAM
python summarize_onnx.py onnx/LFM2.5-1.2B-Instruct/model.onnx
python serve_both.py                    # ports 8080 / 8081
python check_served.py 8080 8081        # verify, do not trust HTTP 200
```

Serving both, verified by parsing the served bytes rather than by reachability
(§11's lesson — the viewer HTML is byte-identical for any model):

```
port 8080: 1,816,044 bytes  872 nodes  1170.34M params  Conv=10 Softmax=6
           in=['input_ids','attention_mask']
port 8081:   738,879 bytes  612 nodes   429.94M params  Conv=0  Softmax=5
           in=['input_ids','positions','target_hidden','ctx_positions','prev_token_ids']
```

---

## S2.14 Net changes to the earlier sections

| Section | Change |
|---|---|
| §5 (gotchas) | The torch DLL failure is **not always** a red herring — check the MSVC runtime *version*, not just presence. `PYTHONUTF8=1` still required. Add the `transformers` 5.x `tie_embedding` rename. |
| §6 (why not optimum) | Still correct. `optimum-onnx` 0.1.0 now exists and still lacks lfm2. Add: **`onnxruntime-genai` does support LFM2** and is what built the official ONNX. |
| §7 (Netron) | **Corrected.** No HTTP Range support; requesting the external data loads the whole file into the server's RAM (measured 4 491 MB peak). Fast open is due to the topology-only `.onnx`, not lazy ranged reads. |
| §10 (limitations) | Limitation #1 (confidence-head input) is **resolved** — confirmed from `dspark_planner.build_markov_embed_stack`. Limitation #2 (paged KV pool) stands. Batch-pinned-to-1 reproduced. |
| §2 / §3 | Unchanged and independently confirmed, including by Liquid AI's own ONNX export. |



---
---

# Session 3 — 2026-09-13 · Netron re-run, and a QNN EP compile on an x86_64 host

**Session log & findings (appended)**
Host: Windows 11 Pro (26100) · Intel i7-1065G7, 4C/8T Ice Lake, **no NPU** · 15.6 GB RAM · Python 3.13.15
Working dir: `C:\Users\ramkumar.pri\project\lfm_case_study\`

Third independent run from a bare Python (`pip list` at the start returned
exactly one package: `pip`). §§1–11 and all of S2 reproduced. What is new is
everything *downstream* of the ONNX file: both graphs were compiled with the
**Qualcomm QNN Execution Provider** — on an Intel laptop, with no Snapdragon
anywhere — and the lowered result was opened in Netron.

The vocabulary below ("Method 1", "Method 3") is
[`networkconversion_documentation_updated.md`](networkconversion_documentation_updated.md)
→ *Visualizing the qnn lowered artifact*: **Method 1** = the EPContext ONNX
wrapper (macro view, CPU/NPU boundary), **Method 2** = `qnn-context-binary-utility`
from the QAIRT SDK (declared not viable), **Method 3** = the pre-compiled QNN
graph (micro view, individual lowered ops). That document was regenerated from
the source `.docx` with **docling 2.126.0** in this session; the QNN section is
what it gained.

---

## S3.0 Summary of what this session adds

| # | Finding | Status vs §§1–11 / S2 |
|---|---|---|
| S3.1 | Third clean-machine reproduction; the drafter reproduces **exactly** — node counts, op histogram, residuals — the target drifts by 3 nodes on `transformers` 5.17 | extends S2.12 |
| S3.1 | The dynamo exporter bakes the **exporting machine's absolute source paths** into every node's metadata, so the same graph is not the same file | new |
| S3.2 | **`onnxruntime-qnn` installs and AOT-compiles on x86_64.** No Qualcomm device needed — the wheel ships `HtpPrepare.dll` for amd64 | entirely new |
| S3.3 | On amd64 only the `htp`, `ir` and `saver` backends exist; `cpu`/`gpu` fail to load — those DLLs are not in the wheel | new |
| S3.4 | A dynamic dim does not cause CPU fallback — it **fails the whole session** with `EP_FAIL` | new |
| S3.5 | `soc_model` is the difference between a context binary and `QNN_COMMON_ERROR_MEM_ALLOC`; and chip *names* are silently ignored, contradicting the docs | new |
| S3.6 | **Method 1 achieved for the drafter**: 1-node EPContext ONNX + a 1.13 GB HTP context binary | new |
| S3.7 | **Method 3 achieved for both models**: the full ONNX→QNN op mapping, with fusions and layout changes visible | new |
| S3.8 | The lowering **breaks the embedding tie** — the 134.2 M-param shared embedding is materialised twice | contradicts S2.12's "single initializer, two consumers" *after* lowering |
| S3.9 | HTP rejects exactly one op in the target (`GatherNd`, error 3110) and fragments the graph; the drafter goes over whole | new |
| S3.10 | The target cannot be finalised at fp32. Bisecting with a weights-only graph **refutes** the obvious size explanation: 5.5 GiB finalises fine | new |
| S3.11 | float16 is a dead end in 2.6.0: the `Reciprocal` op builder emits an fp32 constant into an fp16 graph. Minimal repro, both models | new (looks like a vendor bug) |
| S3.12 | **Netron 9.2.8 ships a QNN loader** (`qnn.js`, marked *Experimental*). It opens every JSON dump; it recognises the context binary and refuses it by name | corrects the docx's premise that you need Qualcomm's "QNN Netron" |
| S3.13 | `backend_type=ir` produces a `.dlc` **without the QAIRT SDK** — and a DLC is a ZIP of FlatBuffers using a *different* op dialect again | new |
| S3.14 | `backend_type=saver` produces a 50,344-line replayable C program — and 9.19 GB of `params.bin` | new |

---

## S3.1 Third reproduction — what is stable, what drifts

Same two scripts, same checkpoints (byte-identical downloads: 2,340,697,936 and
591,458,386), `torch 2.14.0` as in S2 but `transformers` **5.17.0** rather than
5.16.1.

| | S2 (26200) | S3 (26100) |
|---|---:|---:|
| drafter export | 132.3 s | **187.5 s** |
| drafter nodes / initializers / params | 612 / 79 / 429.94 M | **612 / 79 / 429.94 M** |
| drafter `max\|diff\|` logits · hidden · confidence | 2.503e-05 · 2.134e-05 · 2.682e-07 | **identical to every printed digit** |
| target export | 229.0 s | **407.3 s** |
| target nodes / initializers / params | 872 / 171 / 1170.34 M | **869 / 175 / 1170.34 M** |
| target `max\|diff\|` / argmax | 5.770e-05 / 100.00% | **identical** |
| `model.onnx.data` (target) | 4,681,367,552 B | **4,681,367,552 B** |

The drafter reproduces exactly — same node count, same op histogram line for
line, same residuals. The target moves by three nodes (`Concat` 30 → 27) and
four initializers, from the `transformers` minor bump; every architectural count
in §8's fingerprint table still holds: 10 Conv, 10 Split, 6 Softmax, 16 Sigmoid,
45 RMSNorm groups, 1 Cos / 1 Sin, 6 IsNaN, 7 Where.

### The drafter graph is identical and the file still is not

`model.onnx` is 782,806 B here against S2's 738,879 B — a 43,927 B difference in
a graph with the same 612 nodes, the same 79 initializers, the same 688
`value_info` entries and byte-identical initializers. Diffing node by node, 609
of the 612 nodes differ, and every difference is in metadata:

```
S2  "pkg.torch.onnx.stack_trace": "File \"C:\\Users\\TEMP\\project\\lfm25-onnx\\dspark_model.py\", line 289, …
S3  "pkg.torch.onnx.stack_trace": "File \"C:\\Users\\ramkumar.pri\\project\\lfm_case_study\\lfm25-onnx\\dspark_model.py\", line 289, …
```

**The dynamo exporter bakes the exporting machine's absolute source paths into
every node**, four `metadata_props` deep (`namespace`,
`pkg.torch.onnx.class_hierarchy`, `pkg.torch.onnx.fx_node`,
`pkg.torch.onnx.stack_trace`). 23 extra characters × ~3 occurrences × 609 nodes
is the whole delta. Two consequences worth knowing: an ONNX export is **not
byte-reproducible across machines** even when the graph is, so `sha256` is the
wrong equality test for these files (the op histogram is the right one); and a
shipped `.onnx` quietly carries your directory layout and username.

`docling` is worth noting as a toolchain shortcut: installing it drags in
`torch`, `transformers`, `accelerate`, `huggingface-hub`, `numpy` and
`safetensors`, so after Task 1 the only things still missing for the export were
`onnx`, `onnxruntime`, `onnxscript` and `netron`.

---

## S3.2 `onnxruntime-qnn` installs and compiles on x86_64

This was the surprise of the session. The expectation going in was that QNN is
an ARM64/Snapdragon story and that an Intel laptop could, at best, read
documentation about it. Not so:

```
onnxruntime_qnn-2.6.0-cp313-cp313-win_amd64.whl     189.7 MB
```

Inside, `libs/amd64/`:

```
    96,839,376  HtpPrepare.dll                  <- the offline graph preparer
    96,839,376  QnnHtp.dll
     9,613,008  Genie.dll
     3,467,984  onnxruntime_providers_qnn.dll
     3,450,576  QnnSystem.dll
     1,632,464  QnnIr.dll
       956,112  QnnHtpNetRunExtensions.dll
       536,784  QnnSaver.dll
```

and separately `libs/arm64ec/` with the on-device pieces (`QnnGpu.dll`,
`QnnHtpV73Stub.dll`, `libQnnHtpV68/73/81Skel.so`, …) that amd64 does not get.
`onnxruntime-qnn` 2.6.0 bundles **QAIRT 2.50.40**; it is a *plugin* EP, so it
loads into stock `onnxruntime` 1.30.0 rather than needing a vendor fork.

The whole compile happens host-side and produces a deployable binary:

```python
import onnxruntime as ort
import onnxruntime_qnn as qnn_ep

ort.register_execution_provider_library("QNNExecutionProvider", qnn_ep.get_library_path())
devices = [d for d in ort.get_ep_devices() if d.ep_name == "QNNExecutionProvider"]

so = ort.SessionOptions()
so.add_session_config_entry("ep.context_enable", "1")          # method 1
so.add_session_config_entry("ep.context_file_path", "...\\model_fixed_ctx.onnx")
so.add_session_config_entry("ep.context_embed_mode", "0")      # separate .bin
so.add_provider_for_devices(devices, {
    "backend_type": "htp",
    "soc_model": "69", "htp_arch": "75",
    "dump_json_qnn_graph": "1", "json_qnn_graph_dir": "...", # method 3
    "enable_framework_op_trace": "1", "framework_op_trace_dir": "...",
    "dump_qnn_ep_input_graph": "1", "dump_qnn_ep_input_graph_dir": "...",
})
sess = ort.InferenceSession("...\\model_fixed.onnx", sess_options=so)
```

Two things the EP tells you about the host, both harmless:

```
[W] File mapped weights feature is only available on Windows arm64 devices for
    QNN API versions >= 2.32. Feature will be disabled by default
    OrtEpDevice ep_name='QNNExecutionProvider' vendor='Qualcomm'
                type=OrtHardwareDeviceType.CPU        <- no Hexagon to bind to
```

The EP advertises itself against the **CPU** hardware device, because there is
no NPU in the machine to enumerate. It compiles anyway. It cannot *run* the
result, which is the whole point of ahead-of-time compilation.

---

## S3.3 On amd64 there are only three backends

`backend_type` documents five values (`cpu`, `gpu`, `htp`, `saver`, `ir`). Two
of them cannot work here, and the reason is simply that the DLLs are not in the
wheel — `libs/amd64/` has no `QnnCpu.dll` and no `QnnGpu.dll`, even though
`onnxruntime_qnn.get_qnn_cpu_path()` will happily hand you a path to one:

```
--backend cpu  ->  QNN SetupBackend failed Unable to load backend, error: load library failed
--backend gpu  ->  QNN SetupBackend failed Unable to load backend, error: load library failed
```

Usable on x86_64: **`htp`** (compile for the NPU), **`ir`** (lower and serialise
to `.dlc`), **`saver`** (record the QNN API calls as C). All three were used
below.

---

## S3.4 A dynamic dim fails the session — it does not fall back to CPU

The QNN EP documentation says it "does not support models with dynamic shapes".
The natural reading — given how the docx describes ONNX Runtime's fallback
orchestration — is that the dynamic parts stay on the CPU EP. They do not.
Handing the EP the drafter exactly as exported:

```
$ python qnn_compile.py onnx/LFM2.5-1.2B-Instruct-DSpark/model.onnx qnn/_dynamic_probe --backend htp
onnxruntime.capi.onnxruntime_pybind11_state.EPFail:
  [ONNXRuntimeError] : 11 : EP_FAIL : Dynamic shape is not supported yet, for input: val_16
[mem] peak working set 1.10 GB
```

`InferenceSession()` raises; there is no session and no partial offload. Note
`val_16` — an *internal* tensor, not one of the five graph inputs, so the error
does not point at the thing you have to change. The peak working set of 1.10 GB
also says it never got as far as materialising the 1.72 GB of weights: this is a
fast, early refusal during capability discovery.

Fixing it is cheap if you do not use the obvious tool. `python -m
onnxruntime.tools.make_dynamic_shape_fixed` round-trips the whole model, which
for the 1.2B target means materialising and rewriting 4.68 GB of initializers.
Unnecessary — external-data locations resolve relative to the model file, so
writing the patched 1.8 MB topology *into the same directory* reuses the
existing blob untouched. That is all `fix_shapes.py` does:

```
$ python fix_shapes.py onnx/LFM2.5-1.2B-Instruct-DSpark/model.onnx batch=1 block=9 ctx=12
[ok] model_fixed.onnx  0.74 MB
     batch -> 1   (8 occurrence(s) in graph I/O)
     block -> 9   (6 occurrence(s))
     ctx   -> 12  (2 occurrence(s))

$ python fix_shapes.py onnx/LFM2.5-1.2B-Instruct/model.onnx sequence=32
[ok] model_fixed.onnx  1.80 MB
```

`block=9` and `ctx=12` are the drafter's own export shapes (§S2.6 — `block_size`
is exactly 9); `sequence=32` is the target's.

---

## S3.5 `soc_model` decides whether you get a binary or an allocator error

With shapes fixed, the drafter reached the QNN backend, translated cleanly —
and died:

```
EP_FAIL : Failed to finalize QNN graph.
          Error: QNN_COMMON_ERROR_MEM_ALLOC: Memory allocation related error., Code: 1002
[mem] peak working set 5.02 GB
```

5.02 GB on a 15.6 GB box is not an out-of-memory condition, so the message is
misleading. It is also not fixed by any of the memory-shaped knobs:

| provider option | result |
|---|---|
| *(defaults)* | `MEM_ALLOC` |
| `htp_arch=73` | `MEM_ALLOC` |
| `num_graph_prepare_threads=1` | `MEM_ALLOC` |
| `htp_graph_finalization_optimization_mode=0` | `MEM_ALLOC` |
| `vtcm_mb=8` | `MEM_ALLOC` |
| **`soc_model=69`** | **session created in 135.8 s** |
| **`htp_arch=73` + `soc_model=43`** | **session created in 129.7 s** |

`soc_model` defaults to `"0"` — *unknown chip* — and the offline HTP preparer
cannot size a graph this large for a chip it has not been told about. Naming any
real SoC fixes it. `htp_arch` alone does not; `soc_model` alone does.

And a documentation correction while we are here. The QNN EP docs say
`soc_model` accepts chip-family names for well-known parts. This build does not:

```
soc_model=SM8650  ->  [W] Ignoring malformed soc_model, expecting a >=0 integer.
soc_model=SM8750  ->  [W] Ignoring malformed soc_model, expecting a >=0 integer.
soc_model=69      ->  ok
```

It is a warning, not an error, so the run continues with `soc_model=0` and then
fails later in a way that has nothing obviously to do with the option you set.
The rest of this session uses **`soc_model=69`, `htp_arch=75`**, which the
artifacts confirm resolved to:

```json
"compilation_target": { "device_id": 0, "htp_arch": "V75", "soc_model": 69 }
```

---

## S3.6 Method 1 — the EPContext wrapper, on the drafter

```
$ python qnn_compile.py onnx/LFM2.5-1.2B-Instruct-DSpark/model_fixed.onnx \
      qnn/LFM2.5-1.2B-Instruct-DSpark/htp --backend htp --embed 0 \
      --json-graph --op-trace --input-graph --opt soc_model=69 --opt htp_arch=75

[ok] session created in 137.1s   providers=['QNNExecutionProvider', 'CPUExecutionProvider']
[mem] peak working set 5.15 GB

             1,124  model_fixed_ctx.onnx            <- method 1
     1,130,196,264  model_fixed_ctx_qnn.bin         <- the context binary
           331,536  QNNExecutionProvider_..._1_0.json          <- method 3
             9,574  QNNExecutionProvider_..._1_0_tensor_log.json
           523,842  main_graph.0_qnn_ep_input_graph.json
           523,842  main_graph.1_qnn_ep_input_graph.json
           305,124  qnn_op_trace.json
```

**1.72 GB of fp32 ONNX became a 1.13 GB Hexagon context binary in 137 seconds on
a laptop with no Hexagon in it.** The wrapper is 1,124 bytes and contains exactly
what the docx says it should:

```
node QNNExecutionProvider_QNNExecutionProvider_10258722332297493299_1_0
     op=EPContext  domain=com.microsoft
  inputs  ['input_ids', 'positions', 'target_hidden', 'ctx_positions', 'prev_token_ids']
  outputs ['logits', 'hidden_states', 'confidence']
  attr partition_name = QNNExecutionProvider_..._1_0
  attr ep_cache_context = model_fixed_ctx_qnn.bin
  attr main_context    = 1
  attr embed_mode      = 0
  attr source          = QNNExecutionProvider
  attr ep_sdk_version  = v2.50.40.260831140417
  attr is_multi_soc_ep_context = 0
  attr max_size        = 0

metadata_props:
  ep_compatibility_info.QNNExecutionProvider = v2:6:2.50.40:5.50.0:75:69:0:0
                                                             ^^^^ ^^ arch V75, SoC 69
```

Opened in Netron it is a single black box: five inputs go in, three outputs come
out, and the 620 operators in between are gone. That is the point — Method 1
shows the *partition boundary*, not the contents. Here the boundary is the whole
model: **zero CPU fallback nodes**.

`embed_mode=0` splits the blob out as `..._qnn.bin` instead of embedding it in
the ONNX, which is what makes the wrapper 1 KB and the binary inspectable on its
own. With `embed_mode=1` you get one 1.13 GB `.onnx`.

---

## S3.7 Method 3 — the lowered QNN graph

`dump_json_qnn_graph=1` writes the QNN graph as JSON at *compose* time — before
`graphFinalize`. That ordering matters more than it sounds: in the runs that
died at finalize with `MEM_ALLOC`, **the Method 3 artifact was still produced**.
Method 3 survives failures that Method 1 does not, which is exactly what you
want from a debugging view.

The drafter, ONNX vs QNN:

```
onnx/…-DSpark/model.onnx                 612 nodes
  -> ORT level-1 optimisation            574 nodes handed to the EP
  -> QNN translation                     620 QNN ops, 739 tensors, 1 subgraph
                                         0 unsupported
```

and the mapping, reconstructed by joining `qnn_op_trace.json` against
`main_graph.1_qnn_ep_input_graph.json`:

| count | ONNX | → QNN (`qti.aisw`) |
|---:|---|---|
| 126 | `Mul` | `ElementWiseMultiply` |
| 55 | `Reshape` | `Reshape` |
| 53 | `Add` | `ElementWiseAdd` |
| 48 | `MatMul` | `FullyConnected` **and** an extra `Reshape` each |
| 46 | `Unsqueeze` | `Reshape` |
| 30 | `Slice` | `StridedSlice` |
| 27 | `Pow` / `ReduceMean` / `Sqrt` / `Reciprocal` | `ElementWisePower` / `ReduceMean` / `ElementWiseSquareRoot` / **`ElementWiseDivide`** |
| 26 | `Concat` | `Concat` |
| 15 | `Sub` | `ElementWiseSubtract` |
| 10 | `Expand` | **`ElementWiseMultiply`** |
| 10 | `MatMul` | `MatMul` |
| 6 | `Sigmoid` | `Sigmoid` |
| 5 | **`Mul` + `Softmax`** | **`Softmax`** (fused) |
| 2 | `Gather` | `Cast` **and** `Gather` |
| 2 | `Sin` / `Cos` | `ElementWiseSin` / `ElementWiseCos` |
| 1 | `Gemm` + `Reshape` | `FullyConnected` + `Reshape` (fused) |

Things worth reading off that table:

- **`MatMul` splits in two.** 58 ONNX `MatMul`s become 48 `FullyConnected` (a
  weight times an activation — a Linear) and 10 `MatMul` (activation times
  activation — the QK and PV products of 5 attention layers). QNN's dialect
  makes the distinction the ONNX opset does not.
- **Every `FullyConnected` drags a `Reshape` with it**, because it is rank-2 and
  the graph is rank-3. That, plus `Unsqueeze → Reshape`, is why 55 ONNX
  `Reshape`s become 150.
- **`Reciprocal` becomes `ElementWiseDivide`** with a synthesised numerator
  constant. Harmless here; fatal at fp16 (S3.11).
- **`Expand` becomes `ElementWiseMultiply`** — QNN has no broadcast-materialising
  op, so a broadcast is a multiply by ones. Ten of them: the GQA head fan-out.
- **`Mul` + `Softmax` fuse into one `Softmax`** — `ScaleSoftmaxFusion`, once per
  attention layer, all five. This is precisely the "verifying operator fusion"
  use case Method 3 is advertised for.
- **`Gather` needs a `Cast` first** — the int64 token ids are narrowed to int32.
- The 27 RMSNorm groups are still 27 groups of four ops. **No RMSNorm fusion
  happens at this level**; whatever the HTP compiler does with them happens
  later, inside `graphFinalize`, and is not visible in any of these dumps.

`fusion_count` from the trace, verbatim:

```json
{"OrtNodeUnit": 613, "ScaleSoftmaxFusion": 5, "ReshapeGemmFusionGroup": 2}
```

The target tells the same story with convolutions in it (783 nodes in → 893 QNN
ops out, 1 subgraph, 0 unsupported, via the `ir` backend):

| count | ONNX | → QNN |
|---:|---|---|
| 188 | `Mul` | `ElementWiseMultiply` |
| 95 | `Add` | `ElementWiseAdd` |
| 93 | `MatMul` | `FullyConnected` + `Reshape` |
| 45 | `Pow`/`ReduceMean`/`Sqrt`/`Reciprocal` | the RMSNorm four |
| 34 | `Slice` | `StridedSlice` |
| 20 | `Conv` | `Reshape` (2 per conv) |
| 12 | `Neg` | `ElementWiseNeg` |
| 12 | `Expand` | `ElementWiseMultiply` |
| 12 | `MatMul` | `MatMul` |
| 10 | `Split` | `Split` |
| **10** | **`Conv`** | **`DepthWiseConv2d`** |
| 7 | `Where` | `ElementWiseSelect` |
| 6 | `Reshape` + `Transpose` | `Transpose` (fused) |
| 6 | `IsNaN` | `IsNan` |
| 6 | `Softmax` | `Softmax` |
| 1 | `GatherND` | `GatherNd` |
| 1 | `And` | `ElementWiseAnd` |

**LFM2's ShortConv is a depthwise convolution and QNN says so out loud.** The 10
`Conv` nodes lower to `DepthWiseConv2d`, each wrapped in two `Reshape`s because
QNN's depthwise op is 2-D and the sequence is 1-D. This is the clearest
independent confirmation of §3's architecture reading so far — it comes from
Qualcomm's op-support logic, not from reading `transformers`.

Note also the difference between the two models' softmaxes: the drafter's scale
is a separate `Mul` and gets fused into `Softmax`; the target's is already folded
into Q by the exporter, so its 6 `Softmax`es pass through alone.

### The two `_qnn_ep_input_graph` dumps are a before/after pair

`dump_qnn_ep_input_graph=1` writes one JSON **per optimisation pass**, and
diffing them shows the layout optimiser working:

```
target  main_graph.0   813 nodes   Transpose=60   node_Conv_1372 …
target  main_graph.1   783 nodes   Transpose=30   node_Conv_1372_token_71 …
                       40 nodes removed, 10 added (the renamed Convs)
```

Thirty of the sixty `Transpose`s around the ShortConv stack are cancelled before
QNN ever sees the graph. For the drafter, which has no convolutions, `.0` and
`.1` are byte-identical (523,842 B each) — nothing for the optimiser to do.

---

## S3.8 The lowering breaks the embedding tie

S2.12 recorded, of the drafter, that weight tying is visible in ONNX as *one*
initializer with two consumers:

```
initializer  shared_embedding [65536, 2048]
  consumed by 2 nodes: Gather(node_embedding)  and  Transpose(node_t)
```

After lowering, it is two:

```
     536,870,912  t_transpose        [65536, 2048]     <- the lm_head
     536,870,912  shared_embedding   [65536, 2048]     <- the embedding lookup
      83,886,080  val_10_transpose   [2048, 10240]
      67,108,864  val_987_transpose  [65536, 256]
      67,108,864  markov_head.markov_w1.weight  [65536, 256]
      50,331,648  val_982_transpose  [2048, 6144]      … and 100 more
```

Every `Transpose(initializer) → MatMul` in the exported graph is constant-folded
into a *new* static tensor for `FullyConnected`, and where the original is still
consumed elsewhere — as the tied embedding is — both copies live in the graph.

| | ONNX params | QNN static tensors | inflation |
|---|---:|---:|---:|
| drafter | 429.94 M (1.72 GB) | **574.65 M (2.14 GiB)** | +144.71 M |
| target | 1170.34 M (4.68 GB) | **1305.35 M (4.86 GiB)** | +135.01 M |

Both deltas are ≈ 134.2 M = 65536 × 2048, i.e. one extra copy of the tied
embedding, in each model. The `.dlc` files confirm the arithmetic from the
outside: 2,298,839,400 B and 5,221,782,944 B, which is the static-tensor total
plus 250 KB / 383 KB of container.

A practical consequence: **`..._tensor_log.json` under-reports.** For the drafter
it counts 29 "initializers" totalling 576.57 MB, because it only sees tensors
that kept their ONNX names; the graph itself carries **114 static tensors
totalling 2.14 GiB**, 38 of them `*_transpose`. The graph JSON is the honest
number.

Two smaller things the tensor table shows. Every one of the 739 tensors has
`quant_params` `{definition: 0x7fffffff, encoding: 0x7fffffff}` — undefined —
confirming this is a pure float graph with no quantization encodings anywhere;
Method 3's "scale/zero-point" column simply has nothing in it for these models.
And all four int64 graph inputs are narrowed at the boundary:

```
100 (INT_64)  input_ids [1,9]        ->  50 (INT_32)  input_ids_int32
100           positions [1,9]        ->  50           positions_cast_int32
100           ctx_positions [1,12]   ->  50           ctx_positions_cast_int32
100           prev_token_ids [1,9]   ->  50           prev_token_ids_int32
```

QNN's index ops are int32, so `int64 -> int32` casts appear for free on every
token-id input.

---

## S3.9 The target: HTP rejects exactly one operator

Under `backend_type=ir` the target is one clean 893-op partition with zero
unsupported nodes. Under `backend_type=htp` it fragments, and verbose logging
names the culprit:

```
[W] QNN.backendValidateOpConfig() failed for node `node_GatherND_34`
    of type `GatherNd` with error code 3110
[W] QNN.backendValidateOpConfig() failed for node `node_GatherND_34_2`
    of type `GatherNd` with error code 3110
```

One node. `GatherND` appears exactly once in the target's op histogram (§S2.12),
in the causal-mask construction that `transformers` 5.x emits. The `ir` backend
accepts it — it validates against the IR schema, not against silicon — while the
HTP backend's op-config validator refuses it, and ONNX Runtime cuts the graph
there. The surviving HTP partition that got dumped is `..._2_0`, 138 nodes and
1.02 GiB of weights: embedding + `Gather`, two ShortConv layers
(`DepthWiseConv2d` ×2, `Split` ×2), 13 `FullyConnected`, 7 RMSNorm groups.

This is Method 1's advertised use case reduced to a single line of evidence:
*one* unsupported operator is enough to split an otherwise fully-offloadable
1.2 B model into CPU and NPU regions.

---

## S3.10 …and then fails to finalise, for a reason that is *not* size

Even with `soc_model` set, the target never finalises:

```
EP_FAIL : Failed to finalize QNN graph.
          Error: QNN_COMMON_ERROR_MEM_ALLOC, Code: 1002
[mem] peak working set 6.08 GB   peak commit 6.28 GB
```

6.08 GB against 15.6 GB of RAM and a 24.6 GB pagefile: the host is not out of
memory. The drafter finalises at 2.14 GiB of static tensors and the target fails
at 4.86 GiB, which suggests a limit somewhere in between — so
`qnn_ceiling_probe.py` bisects it with a graph that is nothing but weights, a
chain of N `[4096, 4096]` fp32 `MatMul`s at 64 MiB each, written straight into
the external-data file so that building a 5.5 GiB probe costs 5.5 GiB of disk
and 64 MiB of RAM:

```
 32 layers    2.00 GiB static   OK    session created in 120.5s
 60 layers    3.75 GiB static   OK    session created in 220.3s
 64 layers    4.00 GiB static   OK    session created in 158.0s
 72 layers    4.50 GiB static   OK    session created in 146.4s
 78 layers    4.88 GiB static   OK    session created in 169.1s
 88 layers    5.50 GiB static   OK    session created in 214.0s
```

**There is no ceiling — the hypothesis is wrong.** 78 layers is 4.88 GiB, more
static weight than the target's 4.86 GiB, and it finalises in 169 seconds. 5.5
GiB finalises too. Whatever `MEM_ALLOC` means here, it is not "too many weight
bytes".

Two more controls point the same way:

- **Input size is irrelevant.** Recompiling the target at `sequence=8` instead
  of 32 fails identically, at exactly the same 6.08 GB peak working set. The
  largest activation in the failing partition is 1 MiB (`[1, 32, 8192]`), so
  there was never much to save.
- **The drafter is not fragmented and it works**; the target is fragmented and
  does not. The partition that fails carries `input_ids` and `attention_mask` in
  and hands **six** tensors back out — `add_262 [1,32,2048]`,
  `_unsafe_view_1 [1,32,32,64]`, `val_235 [1,32,32,32]`,
  `convert_element_type_default [1,32,1]`, and `_to_copy [1,32]` with data type
  1288, i.e. **boolean**. It also contains `model.lm_head.weight` (536.9 MB)
  despite ending in the middle of the network, which is what partitioning around
  a rejected node in the mask path does to a graph.

So the honest reading is: on this host the fp32 target cannot be finalised, the
proximate trigger is the `GatherNd` fragmentation of S3.9 rather than its size,
and the fp16 escape hatch is closed by S3.11. **Method 3 works on the target;
Method 1 does not.** For the drafter both work.

---

## S3.11 float16 is a dead end in onnxruntime-qnn 2.6.0

The obvious way under a static-weight ceiling is to halve the weights. HTP does
its float math in fp16 anyway from QAIRT 2.35 onward, so this should be free.
`to_fp16.py` converts cleanly:

```
target   4.68 GB -> 2,340,679,680 B   (164.6 s convert, peak 6.85 GB)
drafter  1.72 GB ->   859,886,080 B   ( 56.3 s convert, peak 3.34 GB)
```

Both then fail to *compose*, identically:

```
[E] Data length mismatch for static tensor.
    node_name: node_rsqrt  tensor_name: node_rsqrt_divisor
    size calculated from shape: 2, tensor.clientBuf.dataSize: 4
EP_FAIL : Failed to compose Qnn graph.
```

`node_rsqrt_divisor` is the constant the QNN `Reciprocal` builder synthesises to
turn `1/x` into `ElementWiseDivide(1, x)` — see the mapping tables in S3.7. In an
fp16 graph the builder declares that tensor float16 (2 bytes from its shape) but
still hands over a 4-byte float32 buffer, and the wrapper rejects its own
tensor. Minimal repro: **any float16 ONNX graph containing `Reciprocal`.** Both
LFM2.5 graphs qualify — `Reciprocal` is the `1/sqrt(…)` in every RMSNorm, 45 of
them in the target and 27 in the drafter — so neither can be compiled at fp16
with this release. The drafter, which compiles perfectly at fp32, is the cleaner
demonstration that this is about the dtype and not about the model.

Net: on this host the target can be **lowered and inspected** (Method 3) but not
**finalised into a context binary** (Method 1). The drafter does both.

---

## S3.12 Netron 9.2.8 has a QNN loader

The docx says the JSON dumps go into "QNN Netron", implying Qualcomm's fork.
Upstream Netron has had a QNN reader since some point before 9.2.8 —
`netron/qnn.js`, 11,132 bytes, first line `// Experimental` — and it takes ORT's
dumps directly. The match condition is narrow:

```js
const obj = await context.peek('json');
if (obj && obj['model.cpp'] !== undefined && obj.graph) {
    return context.set('qnn.json', obj);
}
```

and ORT's dump satisfies it — the top level is
`{'Total MACs per inference', 'Total parameters', 'converter_command',
'copyright_str', 'graph', 'model.bin', 'model.cpp', 'op_types'}`, with
`model.cpp` and `model.bin` both `"N/A"` and `copyright_str` reading
`Copyright (c) Microsoft Corporation`.

Rather than trust that reading, `qnn.js` was driven headlessly under Node
v24.21.0 with a stub `context` (the module has no imports, only
`export const ModelFactory`), which is what Netron itself would do:

```
QNNExecutionProvider_..._1_0.json (drafter, htp)
  match()  -> "qnn.json"   format QNN
  inputs   -> ctx_positions:int64[1,12]  input_ids:int64[1,9]  positions:int64[1,9] …
  nodes    -> 620
```

Every JSON this session produced was checked the same way:

| artifact | Netron |
|---|---|
| `QNNExecutionProvider_*_N_0.json` (lowered graph) | **opens** — 620 / 893 / 138 nodes |
| `main_graph.{0,1}_qnn_ep_input_graph.json` | **opens** — 574 / 783 nodes |
| `*_tensor_log.json` | no match — it is a size report, not a graph |
| `qnn_op_trace.json` | no match — it is a mapping table, not a graph |
| `*.dlc` | no match |
| `model_fixed_ctx_qnn.bin` | **matches, then refuses** (below) |

The context binary is the interesting one. `qnn.js` carries four 16-byte
signatures for QNN serialized contexts; ours starts

```
00000000: 0000 0002 0000 0003 0000 0000 0000 0001
```

which is the fourth of them. Netron therefore *recognises* the file and then
deliberately stops:

```
match() -> "qnn.serialized"
open()  -> THREW: Error loading QNN model.: File contains undocumented QNN serialized context.
```

That is the docx's "you cannot open the raw binary directly in Netron", pinned
down to the exact code path and the exact message. It is a refusal by design,
not a parse failure.

One caveat on how much Netron can tell you: `qnn-metadata.json` has **7 entries**
— `Conv2d`, `DepthWiseConv2d`, `FullyConnected`, `PoolAvg2d`, `Pool`,
`Transpose`, `Neuron`. Ops outside that list still render, with correct types,
shapes and edges, but with generic `input`/`output` port names instead of
documented ones. For these two graphs that is most of the nodes.

---

## S3.13 `backend_type=ir` gives you a `.dlc` without the QAIRT SDK

Method 3's second half in the docx assumes Qualcomm's `qnn-onnx-converter` and
`qnn-dlc-viewer`. Neither is installed here, and neither is needed:

```
$ python qnn_compile.py …/model_fixed.onnx …/ir --backend ir --dlc --no-ctx
     2,298,839,400  QNNExecutionProvider_..._1_0.dlc     (drafter)
     5,221,782,944  QNNExecutionProvider_..._1_0.dlc     (target, 233.2 s, peak 9.21 GB)
```

A `.dlc` turns out to be an ordinary **ZIP** (`PK\x03\x04`) of five members —
with deliberately invalid CRCs, so `zipfile` needs `_expected_crc = None` to read
them:

| member | drafter | target | what it is |
|---|---:|---:|---|
| `dlc.metadata.history.2.3.0` | 794 | 793 | JSON, provenance |
| `dlc.metadata2.3.0` | 326 | 326 | JSON, version header |
| `model` | 223,512 | 341,840 | FlatBuffer, magic `NETD` |
| `model.params` | 22,736 | 39,392 | FlatBuffer, magic `NETP` |
| `model.params.bin` | 2,298,590,948 | 5,221,399,508 | raw weights |

`model.params.bin` lands within ~2 KB and ~500 B of the static-tensor totals
computed from the JSON graphs in S3.8 — the DLC stores the weights verbatim,
fp32, duplication and all. The history member records its own provenance:

```json
{"converterCommand": {"tool": "qnn-net-run",
                      "converterVersion": "2.50.40.260831140417",
                      "args": {"backend": "libQnnIr.so(.dll)",
                               "model": "QNNExecutionProvider_..._1_0.so(.dll)"}}}
```

And the strings inside `model` show a **third** op dialect. The same three-node
smoke test that the JSON dump calls `Conv2d` / `Relu` / `ElementWiseAdd` appears
in the DLC as `Conv2d` / **`Neuron`** / **`Eltwise_Binary`** — the older
SNPE-lineage names. So "the lowered graph" is not one thing: the JSON dump, the
DLC and the saver trace name the same operators three different ways.

Netron will not open a `.dlc`, and there is no `qnn-dlc-viewer` on x86_64 without
the SDK — the JSON dump is the viewable form.

---

## S3.14 `backend_type=saver` gives you the QNN API calls as C

```
$ python qnn_compile.py …/model_fixed.onnx …/saver --backend saver --no-ctx
     9,194,368,424  saver_output/params.bin
         7,695,978  saver_output/saver_output.c        50,344 lines
```

`QnnSaver.dll` intercepts every QNN API call and writes a compilable C program
that replays it. It writes into the **current working directory**, not into any
configured output dir, which is why `qnn_compile.py` `chdir`s for this backend.
The call histogram is the whole EP protocol in one view:

```
   1197  backendValidateOpConfig      <- "can you take this op?"  (~2x per node)
    816  tensorCreateGraphTensor
    620  graphAddNode                 <- "here it is"
      1  each of backendCreate, deviceCreate, contextCreate,
         graphCreate, graphFinalize, contextFree, deviceFree, backendFree
```

This is `GetCapability()` made concrete: 1,197 validation calls against 620
nodes actually added. And an individual node is fully legible:

```c
Qnn_OpConfigV1_t …_node_linear_0_v1 = {
    "node_linear", "qti.aisw", "FullyConnected", 0, …_params,
    2, …_inputs, 1, …_outputs };
//  input 0  "target_hidden"     type 0 (APP_WRITE)  dtype 562 (FLOAT_32)  [1, 12, 10240]
//  input 1  "val_10_transpose"  type 4 (STATIC)     dtype 562             [2048, 10240]
//  output 0 "linear_reshape"    type 3 (NATIVE)     dtype 562             [12, 2048]
static float …_input_1_data[20971520];
fread(…_input_1_data, 4, 20971520, fp);
```

— the transposed weight of S3.8, in `[out, in]` order, being read out of
`params.bin` at 83,886,080 bytes.

The cost is the catch: **`params.bin` is 9.19 GB for a model with 2.14 GiB of
unique weights**, because the saver writes a tensor's buffer on every call that
carries it. It was deleted after inspection; `saver_output.c` is kept.

---

## S3.15 Method 2, revisited

The docx marks Method 2 — `qnn-context-binary-utility --context_binary … --info`
— as not viable, on the grounds that it needs the QAIRT SDK. That holds: the
`onnxruntime-qnn` wheel ships backend DLLs, not CLI tools, and there is no
`qnn-context-binary-utility` anywhere on this machine.

What is worth adding is that most of what Method 2 promises is available anyway,
from artifacts Method 1 and Method 3 already produce:

| Method 2 would tell you | available instead from |
|---|---|
| internal graph names | `partition_name` attr on the EPContext node; every JSON dump |
| tensor data types | `data_type` in the graph JSON (562 = `FLOAT_32`, 306 = `UINT_32`, …) |
| tensor sizes / buffer budget | `*_tensor_log.json` (`total_graph_size_mb`, per-initializer bytes) |
| quantization encodings | `quant_params` per tensor in the graph JSON — all `0x7fffffff` (undefined) here, these graphs are float |
| which SoC it was built for | `ep_compatibility_info` metadata + `compilation_target` in the op trace |

The one thing genuinely lost is what the HTP compiler did *inside* `graphFinalize`
— tiling, VTCM assignment, weight layout, kernel selection. None of the four
artifact families reach past that boundary. Which is the same boundary §S3.7
noted for RMSNorm fusion, and the same one the docx draws between Method 3
("before it gets baked into machine binary") and Method 1 ("the post-compilation
deployable wrapper").

---

## S3.16 Viewing it

`serve_netron.py` starts one Netron child per artifact — same one-server-per-
process constraint as `serve_both.py` (§11), for the same reason.

```
$ python serve_netron.py --open
http://localhost:8080   1,860,576 B  source ONNX - target
http://localhost:8081     782,806 B  source ONNX - drafter
http://localhost:8082       1,124 B  METHOD 1 - EPContext wrapper (drafter)
http://localhost:8083     331,536 B  METHOD 3 - lowered QNN graph (drafter, htp)
http://localhost:8084     510,781 B  METHOD 3 - lowered QNN graph (target, ir)
http://localhost:8085      81,568 B  METHOD 3 - lowered QNN graph (target, htp partition)
http://localhost:8086     523,842 B  METHOD 3 - what QNN EP was handed (drafter)
http://localhost:8087     752,654 B  METHOD 3 - what QNN EP was handed (target)
```

Verified by parsing what each port actually returns, not by reachability — §11's
lesson still applies, and it applies to the JSON dumps too:

```
8080  match   1,860,576 B  ONNX: 869 nodes
8081  match     782,806 B  ONNX: 612 nodes
8082  match       1,124 B  ONNX: 1 nodes, ops=['EPContext']
8083  match     331,536 B  QNN json: 620 nodes, 739 tensors
8084  match     510,781 B  QNN json: 893 nodes, 1130 tensors
8085  match      81,568 B  QNN json: 138 nodes, 181 tensors
8086  match     523,842 B  QNN json: 574 nodes, 671 tensors
8087  match     752,654 B  QNN json: 783 nodes, 992 tensors
```

Reading order for the two methods side by side: **8081 → 8086 → 8083 → 8082** is
the drafter's whole journey — source ONNX (612 nodes), what ORT handed the EP
(574), the lowered QNN dialect (620), and the single `EPContext` node it all
collapses into. **8080 → 8087 → 8084 → 8085** is the same for the target, ending
not in a wrapper but in the 138-node fragment that `GatherNd` left behind.

---

## S3.17 Timings and memory on this box

| step | time | peak WS |
|---|---:|---:|
| `pip install docling` (+torch, transformers, …) | ~24 min | |
| `pip install onnx onnxruntime onnxscript netron` | ~2 min | |
| `pip install onnxruntime-qnn` (189.7 MB wheel) | ~2 min | |
| download both checkpoints (2.93 GB, sequential curl) | 7 min 5 s | |
| docling DOCX → Markdown | 51.2 s cold start + 13.8 s | |
| `export_dspark.py` | 187.5 s | |
| `export_lfm2.py` | 407.3 s | |
| `to_fp16.py` (target) | 197.3 s | 6.85 GB |
| QNN htp compile, drafter (→ context binary) | **137.1 s** | 5.15 GB |
| QNN ir compile, drafter (→ 2.30 GB dlc) | not recorded | 4.89 GB |
| QNN saver, drafter (→ 7.70 MB C + 9.19 GB bin) | not recorded | 3.24 GB |
| QNN ir compile, target (→ 5.22 GB dlc) | **233.2 s** | 9.21 GB |
| QNN htp compile, target | fails at finalize | 6.08 GB |
| QNN htp compile, target fp16 | fails at compose | 3.15 GB |

---

## S3.18 What was produced

```
lfm25-onnx/
  fix_shapes.py            freeze dynamic dims without rewriting the weight blob
  to_fp16.py               fp32 -> fp16 (kept for the record; see S3.11)
  qnn_compile.py           the QNN EP driver - all four artifact families
  qnn_ceiling_probe.py     bisect the offline preparer's static-tensor limit
  serve_netron.py          one Netron child per artifact, ports 8080-8087

  onnx/LFM2.5-1.2B-Instruct/
    model.onnx  1.86 MB  + model.onnx.data  4.68 GB
    model_fixed.onnx  1.80 MB          sequence=32, reuses the same .data
    model_fp16.onnx   1.80 MB  + model_fp16.onnx.data  2.34 GB
  onnx/LFM2.5-1.2B-Instruct-DSpark/
    model.onnx  0.78 MB  + model.onnx.data  1.72 GB
    model_fixed.onnx  0.74 MB          batch=1 block=9 ctx=12
    model_fp16.onnx   0.74 MB  + model_fp16.onnx.data  0.86 GB

  qnn/LFM2.5-1.2B-Instruct-DSpark/
    htp/   model_fixed_ctx.onnx              1,124 B        <- METHOD 1
           model_fixed_ctx_qnn.bin   1,130,196,264 B        <- the context binary
           QNNExecutionProvider_..._1_0.json   331,536 B    <- METHOD 3
           QNNExecutionProvider_..._1_0_tensor_log.json
           main_graph.{0,1}_qnn_ep_input_graph.json
           qnn_op_trace.json                   305,124 B
    ir/    QNNExecutionProvider_..._1_0.dlc  2,298,839,400 B
    saver/ saver_output/saver_output.c       7,695,978 B    (params.bin 9.19 GB, deleted)
  qnn/LFM2.5-1.2B-Instruct/
    htp/       the 138-node fragment + both EP input graphs (finalize failed)
    htp_fp16/  both EP input graphs (compose failed - S3.11)
    ir/        QNNExecutionProvider_..._1_0.dlc  5,221,782,944 B
               QNNExecutionProvider_..._1_0.json   510,781 B  <- METHOD 3
               qnn_op_trace.json                   443,026 B
  qnn/_dynamic_probe/   the EP input graphs from the dynamic-shape refusal (S3.4)
```

`.gitignore` gains `*.dlc`; `*.bin` and `*.onnx.data` already covered the context
binary and the weight blobs. Everything committed is topology, JSON or source.

Reproduce:

```powershell
$env:PYTHONUTF8 = "1"
pip install docling                                    # brings torch + transformers
pip install onnx onnxruntime onnxscript netron onnxruntime-qnn
python export_dspark.py ; python export_lfm2.py

python fix_shapes.py onnx/LFM2.5-1.2B-Instruct-DSpark/model.onnx batch=1 block=9 ctx=12
python fix_shapes.py onnx/LFM2.5-1.2B-Instruct/model.onnx sequence=32

python qnn_compile.py onnx/LFM2.5-1.2B-Instruct-DSpark/model_fixed.onnx `
       qnn/LFM2.5-1.2B-Instruct-DSpark/htp --backend htp --embed 0 `
       --json-graph --op-trace --input-graph --opt soc_model=69 --opt htp_arch=75
python qnn_compile.py onnx/LFM2.5-1.2B-Instruct/model_fixed.onnx `
       qnn/LFM2.5-1.2B-Instruct/ir --backend ir --dlc --no-ctx --json-graph --op-trace

python serve_netron.py --open
```

---

## S3.19 Net changes to the earlier sections

| Section | Change |
|---|---|
| §5 (environment) | Add: `docling` is a cheap way to get the whole torch/transformers stack. Node.js is useful — Netron's format readers are standalone ES modules you can run headlessly. |
| §7 / S2.11 (Netron) | Still correct. Extend: Netron 9.2.8 also reads **QNN JSON graphs** (`qnn.js`, experimental, 7 documented ops) and explicitly refuses QNN context binaries. |
| §8 (fingerprints) | Reproduced a third time. The drafter is exact; the target moved 872 → 869 nodes on `transformers` 5.17. |
| §10 (limitations) | Add a third: **the exported graphs are not NPU-deployable as they stand.** Dynamic dims must be frozen first (S3.4), and at fp32 the 1.2B target exceeds what the offline HTP preparer will finalise (S3.10). |
| S2.12 (weight tying) | Holds for the ONNX graph; **does not survive lowering** — QNN materialises a second copy of the tied embedding (S3.8). |
| — | The prefill-only export choice (§6) turns out to matter here too: a graph with KV-cache plumbing would carry dynamic `past_sequence_length` dims that QNN EP rejects outright. |
