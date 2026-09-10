---
library_name: sglang
base_model: LiquidAI/LFM2.5-1.2B-Instruct
license: other
license_name: lfm1.0
license_link: LICENSE
pipeline_tag: text-generation
tags:
- speculative-decoding
- dspark
- lfm2
- draft-model
---

<div align="center">
  <img 
    src="https://cdn-uploads.huggingface.co/production/uploads/61b8e2ba285851687028d395/2b08LKpev0DNEk6DlnWkY.png" 
    alt="Liquid AI" 
    style="width: 100%; max-width: 100%; height: auto; display: inline-block; margin-bottom: 0.5em; margin-top: 0.5em;"
  />
  <div style="display: flex; justify-content: center; gap: 0.5em; margin-bottom: 1em;">
    <a href="https://playground.liquid.ai/"><strong>Try LFM</strong></a> • 
    <a href="https://docs.liquid.ai/lfm/getting-started/welcome"><strong>Docs</strong></a> • 
    <a href="https://leap.liquid.ai/"><strong>LEAP</strong></a> • 
    <a href="https://discord.com/invite/liquid-ai"><strong>Discord</strong></a>
  </div>
</div>

# LFM2.5-1.2B-Instruct-DSpark

**LFM2.5-DSpark** is a family of speculative-decoding draft models that adapt DSpark for the LFM2.5 architecture. 
They allow LFM2.5 models to run faster without degrading quality.

This is a drafter for **[`LiquidAI/LFM2.5-1.2B-Instruct`](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct)**. 
In SGLang, decoding runs about 2× faster. It also runs on-device on Apple silicon through the Metal backend.

Find more information about LFM2.5-DSpark in our [blog post](https://www.liquid.ai/blog/lfm2.5-dspark).

## 🗒️ Model Details

LFM2.5-1.2B-Instruct-DSpark is a DSpark speculative-decoding draft model with the following features:

- **Target model**: [`LiquidAI/LFM2.5-1.2B-Instruct`](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct)
- **Draft parameters**: **295.7M** (BF16)
- **Backbone**: 5 full attention layers, `hidden_size=2048`, `intermediate_size=6144` with SiLU/SwiGLU, GQA with `num_attention_heads=32` / `num_key_value_heads=8`, `head_dim=64`
- **Extra heads**: Markov head (rank 256) + confidence head
- **Block size**: 9
- **Vocabulary**: 65,536

Other models in the LFM2.5-DSpark family:

| Drafter | Target |
|---|---|
| [LFM2.5-1.2B-Instruct-DSpark](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct-DSpark) | [LFM2.5-1.2B-Instruct](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct) |
| [LFM2.5-8B-A1B-DSpark](https://huggingface.co/LiquidAI/LFM2.5-8B-A1B-DSpark) | [LFM2.5-8B-A1B](https://huggingface.co/LiquidAI/LFM2.5-8B-A1B) |
| [LFM2.5-2.6B-DSpark](https://huggingface.co/LiquidAI/LFM2.5-2.6B-DSpark) | [LFM2.5-2.6B](https://huggingface.co/LiquidAI/LFM2.5-2.6B) |

## 📊 Performance

### Benchmarks

Speculative decoding is **exact**: the target verifies every proposed token, so the generated
text is what the target would have produced on its own. See [`LiquidAI/LFM2.5-1.2B-Instruct`](https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct) for performance benchmarks.

### Acceptance

Mean accepted tokens per decoding step, by benchmark (1×H100, batch size 1, greedy decoding).
Higher means more of the draft's proposed block is accepted per target forward pass, so decoding
is faster (at block size 9, the ceiling is 10).

| Benchmark | Accepted tokens / step |
|---|---:|
| MATH-500 | 5.78 |
| GSM8K | 4.25 |
| HumanEval | 5.51 |
| MBPP | 5.41 |
| MT-Bench | 3.11 |
| **Mean** | **4.81** |

### On-device and GPU Inference

| Dataset | Acceptance (of 10\) | Speedup on H100 | Speedup on M4 Max |
| :---- | :---- | :---- | :---- |
| MATH500 | 6.02 | **2.56x**<br/>668 → 1712 tok/s | **2.62x**<br/>140 → 366 tok/s |
| HumanEval | 5.31 | **2.26x**<br/>664 → 1499 tok/s | **2.87x**<br/>136 → 389 tok/s |
| MBPP | 5.52 | **2.37x**<br/>667 → 1578 tok/s | **2.74x**<br/>137 → 375 tok/s |
| GSM8K | 4.34 | **1.67x**<br/>624 → 1041 tok/s | **2.73x**<br/>140 → 381 tok/s |
| MT-Bench | 3.90 | **1.66x**<br/>657 → 1091 tok/s | **1.72x**<br/>137 → 237 tok/s |
| Mean | 5.02 | **2.10x**<br/>656 → 1384 tok/s | **2.54x**<br/>138 → 350 tok/s |

## 🏃 How to run (SGLang)

Requires a build of SGLang with DSpark support for LFM2 targets
([PR #31041](https://github.com/sgl-project/sglang/pull/31041)). Launch the target with the drafter
attached:

```bash
python -m sglang.launch_server \
  --model-path LiquidAI/LFM2.5-1.2B-Instruct \
  --speculative-algorithm DSPARK \
  --speculative-draft-model-path LiquidAI/LFM2.5-1.2B-Instruct-DSpark \
  --speculative-draft-attention-backend flashinfer \
  --disable-radix-cache --mem-fraction-static 0.75 --port 30000
```

Then query the OpenAI-compatible endpoint at `http://localhost:30000/v1`. The block size is read
from the draft's `config.json`; the baseline is the same command without the three
`--speculative-*` flags.

## 📬 Contact

- Got questions or want to connect? [Join our Discord community](https://discord.com/invite/liquid-ai)
- If you are interested in custom solutions with edge deployment, please contact [our sales team](https://www.liquid.ai/contact).

## Citation

```bibtex
@article{liquidAI202626B,
  author  = {Liquid AI},
  title   = {LFM2.5-2.6B: Agents Everywhere},
  journal = {Liquid AI Blog},
  year    = {2026},
  note    = {www.liquid.ai/blog/lfm2-5-2-6b},
}
```

```bibtex
@article{liquidAI2026dspark,
  author = {Liquid AI},
  title = {LFM2.5-DSpark: Up to 3.2x Faster Inference from H100 to MacBook},
  journal = {Liquid AI Blog},
  year = {2026},
  note = {www.liquid.ai/blog/lfm2.5-dspark},
}
```
