"""Core Jordan-Spectral Mixer used in the experiments.

This module intentionally contains only the architectural primitive. The training
script under train/train_jsa_mlx.py contains the Parameter Golf / MLX integration.
"""
from __future__ import annotations

import math

import mlx.core as mx
import mlx.nn as nn


class CastedLinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.weight = nn.Linear(in_dim, out_dim, bias=False).weight.astype(mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        return x @ self.weight.astype(x.dtype).T


class JordanSpectralMixer(nn.Module):
    """Spectral-shift token mixer for autoregressive sequence models.

    Args:
        dim: channel width.
        jsa_rank: number of cosine spectral modes retained.
        local_k: number of causal local shifts to mix explicitly.
        max_seq_len: maximum sequence length for the precomputed basis.
    """

    def __init__(self, dim: int, jsa_rank: int, local_k: int, max_seq_len: int = 1024):
        super().__init__()
        self.dim = dim
        self.jsa_rank = max(int(jsa_rank), 1)
        self.local_k = max(int(local_k), 0)
        self.max_seq_len = max_seq_len
        self.local_weights = mx.zeros((self.local_k, dim), dtype=mx.float32) if self.local_k > 0 else None
        self.spectral_gate = CastedLinear(dim, self.jsa_rank)
        self.output_scale = mx.ones((dim,), dtype=mx.float32)
        pos = mx.arange(self.max_seq_len, dtype=mx.float32)[:, None]
        freq = mx.arange(self.jsa_rank, dtype=mx.float32)[None, :]
        self.basis = mx.cos((math.pi / float(self.max_seq_len)) * (pos + 0.5) * freq)

    def __call__(self, x: mx.array) -> mx.array:
        _bsz, seqlen, _dim = x.shape
        pooled = mx.mean(x, axis=1)

        local = mx.zeros_like(x)
        if self.local_k > 0 and seqlen > 1 and self.local_weights is not None:
            for i in range(min(self.local_k, seqlen - 1)):
                k = i + 1
                shifted = mx.pad(x[:, :-k, :], ((0, 0), (k, 0), (0, 0)))
                local = local + shifted * self.local_weights[i].astype(x.dtype)[None, None, :]

        basis = self.basis[:seqlen].astype(x.dtype)
        spectral_coeffs = x.transpose(0, 2, 1) @ basis
        gate = nn.sigmoid(self.spectral_gate(pooled)).astype(x.dtype)[:, None, :]
        global_mix = ((spectral_coeffs * gate) @ basis.T).transpose(0, 2, 1)
        return (local + global_mix) * self.output_scale.astype(x.dtype)[None, None, :]
