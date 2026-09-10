"""Faithful PyTorch reconstruction of LiquidAI/LFM2.5-1.2B-Instruct-DSpark.

The checkpoint declares ``architectures: ["Lfm2DSparkDraftModel"]`` but ships no
modeling code and no ``auto_map``, so nothing on PyPI can load it.  The class
lives in SGLang:

    python/sglang/srt/models/lfm2_dspark.py -> Lfm2DSparkDraftModel(DSparkDraftModel)
    python/sglang/srt/models/dspark.py      -> DSparkDraftMixin, VanillaMarkov,
                                               DSparkConfidenceHead
    python/sglang/srt/models/dflash.py      -> DFlashDraftModel, DFlashDecoderLayer,
                                               DFlashAttention, DFlashMLP

and the proposal-time glue (which decides what actually feeds the confidence
head) lives in

    python/sglang/srt/speculative/dspark_components/dspark_planner.py
        -> build_markov_embed_stack(), compute_confidence()
    python/sglang/srt/speculative/dspark_components/dspark_draft_sampler.py
        -> DsparkDraftSampler.__call__

Parameter names below match the checkpoint exactly, so
``load_state_dict(strict=True)`` is the correctness check on the reconstruction.

Deviations from SGLang, all numerically neutral:
  * q/k/v and gate/up are kept as the separate projections the checkpoint
    stores.  SGLang fuses them into qkv_proj / gate_up_proj at load time.
  * attention is written out (MatMul/Softmax/MatMul) instead of calling a fused
    kernel, so the graph is legible in Netron.
  * the fused add-norm residual (`norm(x, residual)` returning both) is written
    in its plain algebraic equivalent `h = h + f(norm(h))`.
  * context K/V is concatenated as a plain prefix; SGLang writes it into a
    paged KV pool.  Same math, different memory layout.
"""

from __future__ import annotations

import json
import math
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# primitives
# --------------------------------------------------------------------------
class RMSNorm(nn.Module):
    """sglang.srt.layers.layernorm.RMSNorm (forward_native)."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return x.to(dtype) * self.weight


def rope_cos_sin(positions: torch.Tensor, head_dim: int, base: float) -> tuple:
    """cos/sin for `positions`, matching sglang RotaryEmbedding._compute_cos_sin_cache.

    inv_freq = 1 / base ** (arange(0, head_dim, 2) / head_dim)
    freqs    = outer(positions, inv_freq)          -> [..., head_dim // 2]
    """
    inv_freq = 1.0 / (
        base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )
    freqs = positions.float().unsqueeze(-1) * inv_freq  # [B, T, head_dim//2]
    return freqs.cos(), freqs.sin()


def apply_rope_gptj(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """GPT-J / interleaved RoPE  (`rope_is_neox_style: false`).

    Pairs adjacent channels (x0,x1), (x2,x3), ... rather than halves.
    x: [B, T, H, D]   cos/sin: [B, T, D//2]
    """
    cos = cos.unsqueeze(-2).to(x.dtype)  # [B, T, 1, D//2]
    sin = sin.unsqueeze(-2).to(x.dtype)
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    return torch.stack((o1, o2), dim=-1).flatten(-2)


# --------------------------------------------------------------------------
# blocks
# --------------------------------------------------------------------------
class DSparkAttention(nn.Module):
    """dflash.DFlashAttention, unfused.

    NOTE the attention is BIDIRECTIONAL over the proposed block.  In SGLang,
    `_get_dflash_layer_attention_params` maps a "full_attention" draft layer to
    AttentionType.ENCODER_ONLY unless the config sets `is_causal` -- and this
    config does not.  DSpark proposes all `block_size` tokens in one pass, so
    block positions see each other in both directions, plus the whole context
    prefix.
    """

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        h = cfg["hidden_size"]
        self.num_heads = cfg["num_attention_heads"]
        self.num_kv_heads = cfg["num_key_value_heads"]
        self.head_dim = cfg.get("head_dim", h // self.num_heads)
        self.n_rep = self.num_heads // self.num_kv_heads
        self.scaling = self.head_dim**-0.5
        self.rope_theta = cfg["rope_theta"]
        eps = cfg["rms_norm_eps"]

        q_size = self.num_heads * self.head_dim
        kv_size = self.num_kv_heads * self.head_dim
        self.q_proj = nn.Linear(h, q_size, bias=False)
        self.k_proj = nn.Linear(h, kv_size, bias=False)
        self.v_proj = nn.Linear(h, kv_size, bias=False)
        self.o_proj = nn.Linear(q_size, h, bias=False)
        # per-head Q/K RMSNorm over head_dim -- Qwen3's signature
        self.q_norm = RMSNorm(self.head_dim, eps=eps)
        self.k_norm = RMSNorm(self.head_dim, eps=eps)

    def _kv_ctx(self, ctx_hidden: torch.Tensor, ctx_positions: torch.Tensor):
        """DFlashAttention.kv_proj_only + apply_k_norm + apply_k_rope.

        The context prefix never needs Q, so only K/V are projected.
        """
        B, C, _ = ctx_hidden.shape
        k = self.k_proj(ctx_hidden).view(B, C, self.num_kv_heads, self.head_dim)
        v = self.v_proj(ctx_hidden).view(B, C, self.num_kv_heads, self.head_dim)
        k = self.k_norm(k)
        cos, sin = rope_cos_sin(ctx_positions, self.head_dim, self.rope_theta)
        k = apply_rope_gptj(k, cos, sin)
        return k, v

    def forward(self, hidden, positions, ctx_hidden, ctx_positions):
        B, T, _ = hidden.shape
        q = self.q_proj(hidden).view(B, T, self.num_heads, self.head_dim)
        k = self.k_proj(hidden).view(B, T, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden).view(B, T, self.num_kv_heads, self.head_dim)

        q = self.q_norm(q)
        k = self.k_norm(k)
        cos, sin = rope_cos_sin(positions, self.head_dim, self.rope_theta)
        q = apply_rope_gptj(q, cos, sin)
        k = apply_rope_gptj(k, cos, sin)

        k_ctx, v_ctx = self._kv_ctx(ctx_hidden, ctx_positions)
        k = torch.cat([k_ctx, k], dim=1)  # [B, C+T, kvH, D]
        v = torch.cat([v_ctx, v], dim=1)

        # GQA fan-out 8 -> 32
        k = k.repeat_interleave(self.n_rep, dim=2)
        v = v.repeat_interleave(self.n_rep, dim=2)

        q = q.transpose(1, 2)  # [B, H, T,   D]
        k = k.transpose(1, 2)  # [B, H, C+T, D]
        v = v.transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scaling
        # no causal mask: ENCODER_ONLY over the block
        probs = torch.softmax(scores, dim=-1)
        out = torch.matmul(probs, v)  # [B, H, T, D]
        out = out.transpose(1, 2).reshape(B, T, self.num_heads * self.head_dim)
        return self.o_proj(out)


class DSparkMLP(nn.Module):
    """dflash.DFlashMLP -- SwiGLU, silu only."""

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        h, i = cfg["hidden_size"], cfg["intermediate_size"]
        self.gate_proj = nn.Linear(h, i, bias=False)
        self.up_proj = nn.Linear(h, i, bias=False)
        self.down_proj = nn.Linear(i, h, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DSparkDecoderLayer(nn.Module):
    """dflash.DFlashDecoderLayer (no grouped conv: conv_kernel_size unset)."""

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        h, eps = cfg["hidden_size"], cfg["rms_norm_eps"]
        self.input_layernorm = RMSNorm(h, eps=eps)
        self.self_attn = DSparkAttention(cfg)
        self.post_attention_layernorm = RMSNorm(h, eps=eps)
        self.mlp = DSparkMLP(cfg)

    def forward(self, hidden, positions, ctx_hidden, ctx_positions):
        hidden = hidden + self.self_attn(
            self.input_layernorm(hidden), positions, ctx_hidden, ctx_positions
        )
        hidden = hidden + self.mlp(self.post_attention_layernorm(hidden))
        return hidden


class VanillaMarkov(nn.Module):
    """dspark.VanillaMarkov -- rank-256 factorisation of a 65536x65536 bigram
    transition matrix, added as a bias onto the target head's logits."""

    def __init__(self, vocab_size: int, markov_rank: int) -> None:
        super().__init__()
        self.markov_w1 = nn.Embedding(vocab_size, markov_rank)
        self.markov_w2 = nn.Linear(markov_rank, vocab_size, bias=False)

    def get_prev_embeddings(self, token_ids):
        return self.markov_w1(token_ids)

    def project_bias(self, latent):
        return self.markov_w2(latent)


class DSparkConfidenceHead(nn.Module):
    """dspark.DSparkConfidenceHead.

    Input width 2304 = hidden_size 2048 + markov_rank 256.  The second half is
    markov_w1(prev_token) -- confirmed by dspark_planner.build_markov_embed_stack,
    which builds prev_seq = cat([anchor, draft_tokens[:, :gamma-1]]) and passes
    markov_head.get_prev_embeddings(prev_seq) straight into the head.
    """

    def __init__(self, hidden_size: int, markov_rank: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_size + markov_rank, 1, bias=True)

    def forward(self, hidden, markov_embed_stack):
        feats = torch.cat([hidden, markov_embed_stack], dim=-1)
        # apply_sts: sigmoid(raw / sts_temperatures); sts_temperatures is a
        # non-persistent buffer defaulting to 1.0 and is absent from the
        # checkpoint, so at rest this is a plain sigmoid.
        return torch.sigmoid(self.proj(feats).squeeze(-1))


# --------------------------------------------------------------------------
# the model
# --------------------------------------------------------------------------
class Lfm2DSparkDraftModel(nn.Module):
    """DSparkDraftModel = DSparkDraftMixin + DFlashDraftModel."""

    def __init__(self, cfg: dict) -> None:
        super().__init__()
        self.cfg = cfg
        h = cfg["hidden_size"]
        eps = cfg["rms_norm_eps"]
        dfc = cfg["dflash_config"]
        self.block_size = cfg["block_size"]
        self.target_layer_ids = dfc["target_layer_ids"]
        self.num_context_features = len(self.target_layer_ids)
        self.mask_token_id = dfc["mask_token_id"]

        self.layers = nn.ModuleList(
            [DSparkDecoderLayer(cfg) for _ in range(cfg["num_hidden_layers"])]
        )
        self.norm = RMSNorm(h, eps=eps)
        # fc: [num_context_features * hidden] -> hidden.  10240 = 5 x 2048.
        self.fc = nn.Linear(self.num_context_features * h, h, bias=False)
        self.hidden_norm = RMSNorm(h, eps=eps)
        self.markov_head = VanillaMarkov(cfg["vocab_size"], cfg["markov_rank"])
        self.confidence_head = DSparkConfidenceHead(h, cfg["markov_rank"])

        # injected from the target by attach_shared_modules(); NOT in this
        # checkpoint.  LFM2.5-1.2B-Instruct ties them, so one matrix serves as
        # both embedding and head.
        self.register_buffer(
            "shared_embedding", torch.zeros(cfg["vocab_size"], h), persistent=False
        )

    def attach_shared_embedding(self, weight: torch.Tensor) -> None:
        self.shared_embedding = weight.to(torch.float32)

    def forward(self, input_ids, positions, target_hidden, ctx_positions, prev_token_ids):
        # DFlashDraftModel.project_target_hidden
        ctx_hidden = self.hidden_norm(self.fc(target_hidden))

        # DSparkDraftMixin.forward_embed -- shared target embedding
        hidden = F.embedding(input_ids, self.shared_embedding)

        for layer in self.layers:
            hidden = layer(hidden, positions, ctx_hidden, ctx_positions)
        hidden = self.norm(hidden)

        # compute_base_logits: project through the (tied) target lm_head
        base_logits = torch.matmul(hidden, self.shared_embedding.t())

        # VanillaMarkov.apply_block_logits
        markov_embed = self.markov_head.get_prev_embeddings(prev_token_ids)
        logits = base_logits + self.markov_head.project_bias(markov_embed)

        confidence = self.confidence_head(hidden, markov_embed)
        return logits, hidden, confidence


# --------------------------------------------------------------------------
def load_dspark(model_dir: str, target_embedding: Optional[torch.Tensor] = None):
    """Build the module, strict-load the checkpoint, attach the shared head."""
    from safetensors.torch import load_file

    with open(os.path.join(model_dir, "config.json"), "r", encoding="utf-8") as fh:
        cfg = json.load(fh)

    model = Lfm2DSparkDraftModel(cfg)
    sd = load_file(os.path.join(model_dir, "model.safetensors"))
    sd = {k: v.to(torch.float32) for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # shared_embedding is non-persistent and injected, so it is the only
    # legitimate "missing" entry.
    missing = [m for m in missing if m != "shared_embedding"]
    if missing or unexpected:
        raise RuntimeError(
            f"state_dict mismatch: missing={missing} unexpected={unexpected}"
        )
    if target_embedding is not None:
        model.attach_shared_embedding(target_embedding)
    model.eval()
    return model, cfg, len(sd)
