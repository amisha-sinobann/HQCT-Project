"""
Hybrid Quantum-Classical Transformer models.

Implements:
- Classical multi-head attention (baseline)
- HQCT-K  : Quantum Kernel Attention
- HQCT-P  : Quantum Projection Attention
"""

from typing import Optional, Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .quantum_circuits import QuantumKernel, QuantumProjection


class ClassicalMultiHeadAttention(nn.Module):
    """Standard scaled dot-product multi-head attention."""

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, N, _ = x.shape
        q = self.W_q(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        k = self.W_k(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        v = self.W_v(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, N, self.d_model)
        out = self.W_o(out)

        if return_attn:
            return out, attn
        return out, None


class QuantumKernelAttention(nn.Module):
    """
    HQCT-K: classical Q/K/V projections + quantum kernel similarity.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_qubits: int = 6,
        n_layers: int = 3,
        dropout: float = 0.1,
        noise: float = 0.0,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.n_qubits = n_qubits

        # Classical projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        # Project head dimension → n_qubits for the quantum circuit
        self.q_to_qubits = nn.Linear(self.d_k, n_qubits)
        self.k_to_qubits = nn.Linear(self.d_k, n_qubits)

        self.quantum_kernel = QuantumKernel(
            n_qubits=n_qubits, n_layers=n_layers, noise=noise
        )
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(n_qubits)  # analogous to √d_k

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, N, _ = x.shape

        q = self.W_q(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        k = self.W_k(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        v = self.W_v(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)

        # Map to qubit features
        q_feat = self.q_to_qubits(q)  # (B, H, N, n_qubits)
        k_feat = self.k_to_qubits(k)

        # Quantum kernel matrix
        # NOTE: this is O(B·H·N²) circuit evaluations – expensive but exact
        # as described in the paper. For larger N one would use approximations.
        kappa = self.quantum_kernel(q_feat, k_feat)  # (B, H, N, N)
        scores = kappa / self.scale

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, N, self.d_model)
        out = self.W_o(out)

        if return_attn:
            return out, attn
        return out, None


class QuantumProjectionAttention(nn.Module):
    """
    HQCT-P: quantum circuits replace the classical W_Q and W_K projections.
    Value projection stays classical.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_qubits: int = 6,
        n_layers: int = 3,
        dropout: float = 0.1,
        noise: float = 0.0,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Quantum projections for Q and K
        self.q_proj = QuantumProjection(
            d_model, n_qubits=n_qubits, n_layers=n_layers, noise=noise
        )
        self.k_proj = QuantumProjection(
            d_model, n_qubits=n_qubits, n_layers=n_layers, noise=noise
        )
        # Classical value + output
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        # After quantum projection we have vectors of size out_dim.
        # Map them to the full model dimension, then split into heads.
        q_out_dim = self.q_proj.out_dim
        self.q_post = nn.Linear(q_out_dim, d_model)
        self.k_post = nn.Linear(q_out_dim, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attn: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, N, _ = x.shape

        # Quantum features → classical head space
        q_feat = self.q_proj(x)  # (B, N, out_dim)
        k_feat = self.k_proj(x)

        q = self.q_post(q_feat).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_post(k_feat).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        v = self.W_v(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, N, self.d_model)
        out = self.W_o(out)

        if return_attn:
            return out, attn
        return out, None


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1,
        attn_type: str = "classical",
        n_qubits: int = 6,
        n_layers: int = 3,
        noise: float = 0.0,
    ):
        super().__init__()
        if attn_type == "classical":
            self.attn = ClassicalMultiHeadAttention(d_model, n_heads, dropout)
        elif attn_type == "hqct-k":
            self.attn = QuantumKernelAttention(
                d_model, n_heads, n_qubits, n_layers, dropout, noise
            )
        elif attn_type == "hqct-p":
            self.attn = QuantumProjectionAttention(
                d_model, n_heads, n_qubits, n_layers, dropout, noise
            )
        else:
            raise ValueError(f"Unknown attn_type: {attn_type}")

        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None, return_attn=False):
        attn_out, attn_weights = self.attn(self.norm1(x), mask, return_attn)
        x = x + self.dropout(attn_out)
        x = x + self.ff(self.norm2(x))
        return x, attn_weights


class HQCT(nn.Module):
    """
    Hybrid Quantum-Classical Transformer.

    Parameters match the paper (Section 4.2):
      - 4 layers, d_model=256, 4 heads, d_ff=512
      - quantum: 6 qubits, depth 3
    """

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
        d_ff: int = 512,
        max_seq_len: int = 256,
        dropout: float = 0.1,
        attn_type: str = "hqct-k",  # "classical" | "hqct-k" | "hqct-p"
        n_qubits: int = 6,
        q_layers: int = 3,
        noise: float = 0.0,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.attn_type = attn_type
        self.pad_idx = pad_idx

        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    n_heads,
                    d_ff,
                    dropout,
                    attn_type=attn_type,
                    n_qubits=n_qubits,
                    n_layers=q_layers,
                    noise=noise,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_attn: bool = False,
    ):
        B, N = input_ids.shape
        positions = torch.arange(N, device=input_ids.device).unsqueeze(0).expand(B, -1)

        x = self.token_emb(input_ids) + self.pos_emb(positions)
        x = self.dropout(x)

        # Create padding mask for attention (1 = keep, 0 = mask)
        if attention_mask is None:
            attention_mask = (input_ids != self.pad_idx).long()
        # Expand to (B, 1, 1, N) for broadcasting over heads & queries
        mask = attention_mask.unsqueeze(1).unsqueeze(2)

        all_attns = []
        for layer in self.layers:
            x, attn = layer(x, mask=mask, return_attn=return_attn)
            if return_attn:
                all_attns.append(attn)

        x = self.norm(x)
        # Mean pooling over non-padding tokens
        mask_f = attention_mask.unsqueeze(-1).float()
        pooled = (x * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1e-9)
        logits = self.classifier(pooled)

        if return_attn:
            return logits, all_attns
        return logits

    def count_classical_parameters(self) -> int:
        """Count only classical (non-quantum) parameters."""
        total = 0
        for name, p in self.named_parameters():
            if "quantum" not in name and "weights" not in name:
                # crude filter – quantum weights live inside QuantumKernel / QuantumProjection
                total += p.numel()
            else:
                # still count the classical pre/post maps
                if any(k in name for k in ["pre_map", "q_to_qubits", "k_to_qubits", "q_post", "k_post", "W_", "token_emb", "pos_emb", "classifier", "norm", "ff", "net"]):
                    total += p.numel()
        return total

    def count_all_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
