# LFM2.5 case study — ONNX export, Netron inspection, and a data-flow visualiser

Two Liquid AI checkpoints taken apart, converted to ONNX, verified numerically,
and then turned into an interactive visualisation of the tensor moving through
the graph.

| | **LFM2.5-1.2B-Instruct** | **LFM2.5-1.2B-Instruct-DSpark** |
|---|---|---|
| `architectures` | `Lfm2ForCausalLM` | `Lfm2DSparkDraftModel` |
| Params | 1 170 340 608 (1.17 B) | 295 725 953 (295.73 M) |
| Layers | 16 — **hybrid**: 10 ShortConv + 6 attention | 5 attention + Markov + confidence heads |
| Loadable by `transformers` | yes | **no** — the class lives in SGLang |
| Role | the model that generates | drafts 9 tokens/step for the model above |

**Headline finding: these are not two variants of the same model.** The naming
implies a sibling pair; they are a *target* model and its *speculative-decoding
drafter*, with completely different architectures. Full write-up in
[`onnx_session.md`](onnx_session.md).

---

## Repository layout

```
onnx_session.md              Full session log — architecture findings, gotchas,
                             op-histogram fingerprints, corrections. Start here.

networkconversion_documentation_updated.md
                             Edge deployment / network conversion study: how
                             Renesas R-Car V3H (IMPC command lists, tiling,
                             register banks), TI edgeai-tidl-tools, Qualcomm
                             QNN EP and Apache TVM each take a trained network
                             down to hardware — and where the compilation
                             boundary sits in each.

lfm25-onnx/
  dspark_model.py            PyTorch reconstruction of the DSpark drafter.
                             Nothing on PyPI can load this checkpoint; the
                             module is rebuilt op-for-op from SGLang's source
                             and validated by load_state_dict(strict=True).
  export_lfm2.py             Target  → ONNX (prefill graph, fp32) + ORT verify
  export_dspark.py           Drafter → ONNX + ORT verify
  summarize_onnx.py          Graph statistics / op histogram
  serve_both.py              Netron server, one child process per model
  check_served.py            Verifies what each port *actually* serves

  models/                    config.json, tokenizer, README for both models
                             (weights excluded — see below)
  onnx/                      Exported graphs. model.onnx (topology) committed;
                             model.onnx.data (weights) excluded.
  official_onnx/             Liquid AI's own ONNX export, topology only —
                             used as an independent cross-check
  ref/                       SGLang / onnxruntime-genai / optimum-onnx sources
                             consulted while reconstructing the drafter
  logs/

space-pinball/
  index.html                 NEBULA DRIFT — the original space pinball game
  neural-pinball.html        TENSOR DRIFT — the data-flow visualiser built on it
```

---

## What is not in this repository

Four files, 8.69 GB in total, are excluded by `.gitignore`. Each is far over
GitHub's hard 100 MB per-file limit, and Git LFS's free tier is 1 GB, so they
cannot be pushed here.

| File | Size |
|---|---:|
| `onnx/LFM2.5-1.2B-Instruct/model.onnx.data` | 4.36 GB |
| `models/LFM2.5-1.2B-Instruct/model.safetensors` | 2.18 GB |
| `onnx/LFM2.5-1.2B-Instruct-DSpark/model.onnx.data` | 1.60 GB |
| `models/LFM2.5-1.2B-Instruct-DSpark/model.safetensors` | 0.55 GB |

Everything else — every script, config, log, reference source, and the two ONNX
**topology** files — is committed, so the graphs can still be inspected, diffed
and reviewed without downloading a single byte of weights.

### Regenerating the weights

```bash
pip install torch transformers onnx onnxruntime onnxscript \
            "huggingface_hub[hf_xet]" accelerate netron numpy

# checkpoints (2.93 GB)
huggingface-cli download LiquidAI/LFM2.5-1.2B-Instruct        --local-dir lfm25-onnx/models/LFM2.5-1.2B-Instruct
huggingface-cli download LiquidAI/LFM2.5-1.2B-Instruct-DSpark --local-dir lfm25-onnx/models/LFM2.5-1.2B-Instruct-DSpark

# exports (~4 min + ~2 min, ~5 GB RAM for the 1.2B)
export PYTHONUTF8=1              # the dynamo exporter prints a ✅ that cp1252 cannot encode
cd lfm25-onnx
python export_lfm2.py
python export_dspark.py
```

Both scripts verify their own output against PyTorch through ONNX Runtime and
print the residuals. Expected:

```
LFM2.5-1.2B-Instruct   max|diff| 5.770e-05   argmax agreement 100.00%
DSpark  logits 2.503e-05 · hidden 2.134e-05 · confidence 2.682e-07
        strict load: 62 checkpoint tensors, 0 missing, 0 unexpected
```

---

## Viewing the graphs in Netron

```bash
cd lfm25-onnx
python serve_both.py          # http://localhost:8080  and  :8081
python check_served.py 8080 8081
```

`check_served.py` exists because a reachability check is not enough — Netron's
viewer HTML is byte-identical for every model, so a port serving the *wrong*
graph still answers HTTP 200. It parses the served bytes instead:

```
port 8080: 872 nodes  1170.34M params  Conv=10 Softmax=6   -> LFM2 hybrid
port 8081: 612 nodes   429.94M params  Conv=0  Softmax=5   -> DSpark drafter
```

> **Keep `model.onnx` and `model.onnx.data` in the same folder.** Netron
> resolves external data relative to the model file. Also note Netron's Python
> server has no HTTP Range support — requesting the external data pulls the
> whole file into the server process (measured: 4 491 MB peak). Browse topology
> freely; use the Netron desktop app to inspect weight *values*.

---

## TENSOR DRIFT — the data-flow visualiser

`space-pinball/neural-pinball.html` — a single file, no dependencies, no build.
Open it in a browser.

The **ball is the activation tensor**, the **graph is the pinball machine**.
The tensor travels the real execution order and slams into each node; the node
fires, transforms the data and kicks it onward.

The graph is rendered as a faithful Netron view — colours, node anatomy and
geometry taken from Netron 9.2.6's own `grapher.css` (dark theme: canvas
`#404040`, node body `#2d2d2d`, edges `#888`, `layer` headers
`rgba(51,85,136,.7)`, `attention` `rgba(100,50,0,.7)`, …). Nodes show their
initializers as `name <shape>` rows and every edge is labelled with its tensor
shape, exactly as Netron does.

Every shape and parameter count on screen is real: summing the per-node
parameters reproduces **1 170 340 608** and **295 725 953** exactly.

| | |
|---|---|
| `Space` | play / pause |
| `→` | step one node |
| `R` / `F` | restart / fit |
| `A` `B` `C` | Model A · Model B · Compare |
| `V` | data-flow ⇄ architecture view |
| drag · wheel · click | pan · zoom · inspect a node |

`?play&model=B&at=11` deep-links straight to a given node.

In **Compare** mode both models propagate side by side, and the drafter's
shallower path is visible as actual speed — it completes in 716 frames against
the target's 1164.

---

## Sources

- [`LiquidAI/LFM2.5-1.2B-Instruct`](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct)
- [`LiquidAI/LFM2.5-1.2B-Instruct-DSpark`](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct-DSpark)
- [`LiquidAI/LFM2.5-1.2B-Instruct-ONNX`](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct-ONNX) — Liquid AI's official export, built with `onnxruntime-genai`
- SGLang `srt/models/{lfm2_dspark,dspark,dflash}.py` and
  `srt/speculative/dspark_components/*` — the only implementation of the drafter
