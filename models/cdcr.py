"""CDCR main model.

Architecture
------------
- Shared content/twin/style encoder (BERT/RoBERTa)
- Frozen CDG-node encoder (same backbone, no_grad) for typed-edge attention
- Typed-edge attention over CDG (4 edge types: seq, causal, contra, support)
  implemented as masked multi-head attention with edge-type embeddings,
  avoiding PyG dependency.
- Reliability-gated fusion of [h_text, h_cdg, delta] with sample-level
  reliability scalars (rho, DM_norm, Gini) as gate conditioning.
- Loss = L_cls + alpha * L_SCCL + beta * L_tamper + gamma * L_style.

Implementation notes
--------------------
- Designed for PyTorch 1.13 (no torch.compile, no SDPA flash) to match the
  existing FACTGUARD conda env.
- No batching tricks beyond padding; works on a single 24-40GB GPU.
"""
from __future__ import annotations
import math
from typing import Optional, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from .sccl_loss import SCCLLoss, TamperMagnitudeLoss, style_invariance_loss


EDGE_TYPES = 4  # 0=seq, 1=causal, 2=contra, 3=support


# ── Typed-edge attention (no PyG) ────────────────────────────────────────────

class TypedEdgeAttention(nn.Module):
    """One layer of relation-aware attention over a batched padded graph.

    Inputs (per sample):
        x:           [N, D]  node features
        edge_index:  [E, 2]  (src, dst) with -1 padding
        edge_type:   [E]     0..3 with -1 padding
        edge_weight: [E]
        node_mask:   [N]     True for valid
    """

    def __init__(self, dim: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.h = heads
        self.dk = max(1, dim // heads)
        self.qkv = nn.Linear(dim, self.dk * heads * 3)
        self.edge_emb = nn.Embedding(EDGE_TYPES, self.dk * heads)
        self.out = nn.Linear(self.dk * heads, dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.dk)

    def forward(self, x, edge_index, edge_type, edge_weight, node_mask):
        B, N, D = x.shape
        E = edge_index.size(1)
        H, dk = self.h, self.dk

        qkv = self.qkv(x).view(B, N, 3, H, dk)
        q, k, v = qkv.unbind(dim=2)                 # [B, N, H, dk]
        e_type = edge_type.clamp(min=0)             # [B, E]
        e_emb = self.edge_emb(e_type)               # [B, E, H*dk]
        e_emb = e_emb.view(B, E, H, dk)
        e_k = e_emb  # bias on key

        # Clamp to valid indices to avoid gather errors on padding.
        s_idx = edge_index[..., 0].clamp(min=0, max=N - 1)
        d_idx = edge_index[..., 1].clamp(min=0, max=N - 1)
        valid_edge = (edge_index[..., 0] >= 0) & node_mask.gather(1, s_idx) & node_mask.gather(1, d_idx)

        # Gather per-edge (q at dst, k at src, v at src)
        q_e = q.gather(1, d_idx[..., None, None].expand(-1, -1, H, dk))   # [B, E, H, dk]
        k_src = k.gather(1, s_idx[..., None, None].expand(-1, -1, H, dk))
        v_src = v.gather(1, s_idx[..., None, None].expand(-1, -1, H, dk))
        k_e = k_src + e_k

        # [B, E, H]
        score = (q_e * k_e).sum(dim=-1) * self.scale
        # weight
        score = score * edge_weight[..., None].clamp(min=0)
        score = score.masked_fill(~valid_edge[..., None], -1e9)

        # Scatter-softmax over destination index.
        attn = _scatter_softmax(score, d_idx, N)    # [B, E, H]
        attn = attn.masked_fill(~valid_edge[..., None], 0.0)

        # Message: attn * v_src → [B, E, H, dk]
        msg = (attn[..., None] * v_src)
        agg = torch.zeros(B, N, H, dk, device=x.device, dtype=msg.dtype)
        agg.scatter_add_(1, d_idx[..., None, None].expand(-1, -1, H, dk), msg)

        agg = agg.reshape(B, N, H * dk)
        out = self.out(self.dropout(agg))
        return out


def _scatter_softmax(score: torch.Tensor, index: torch.Tensor,
                     dim_size: int) -> torch.Tensor:
    """Stable scatter softmax: for each destination, softmax over its in-edges."""
    B, E, H = score.shape
    big_neg = torch.finfo(score.dtype).min / 2
    # per-destination max for stability
    max_buf = torch.full((B, dim_size, H), big_neg, device=score.device,
                         dtype=score.dtype)
    max_buf.scatter_reduce_(1, index[..., None].expand(-1, -1, H),
                            score, reduce='amax', include_self=False)
    gathered_max = max_buf.gather(1, index[..., None].expand(-1, -1, H))
    exp = torch.exp(score - gathered_max)
    exp = torch.nan_to_num(exp, nan=0.0, posinf=0.0, neginf=0.0)
    exp = exp * (score > big_neg / 2).to(exp.dtype)  # zero out fully-masked edges
    denom_buf = torch.zeros(B, dim_size, H, device=score.device,
                            dtype=score.dtype)
    denom_buf.scatter_add_(1, index[..., None].expand(-1, -1, H), exp)
    gathered_denom = denom_buf.gather(1, index[..., None].expand(-1, -1, H))
    return exp / (gathered_denom + 1e-8)


# ── Main model ──────────────────────────────────────────────────────────────

class CDCR(nn.Module):
    """CDCR: text + CDG + reliability-gated fusion."""

    def __init__(self,
                 encoder_path: str = 'roberta-base',
                 num_labels: int = 2,
                 dropout: float = 0.2,
                 alpha: float = 0.3,
                 beta: float = 0.1,
                 gamma: float = 0.05,
                 alpha_aux: float = 0.4,
                 beta_aux: float = 0.159,
                 tau: float = 0.07,
                 w_hard: float = 2.0,
                 tamper_m: float = 0.5,
                 tamper_eps: float = 0.1,
                 gnn_layers: int = 2,
                 use_nli: bool = True,
                 use_twin: bool = True,
                 use_style: bool = True,
                 use_gate: bool = True,
                 use_aux: bool = True,
                 freeze_bert_layers: int = 0,
                 class_weights=None,
                 use_cdg: bool = True):
        super().__init__()
        self.alpha, self.beta, self.gamma = alpha, beta, gamma
        self.alpha_aux, self.beta_aux = alpha_aux, beta_aux
        self.use_twin = use_twin
        self.use_style = use_style
        self.use_gate = use_gate
        self.use_aux = use_aux
        self.use_cdg = use_cdg

        # Encoders: content encoder (trainable) + node encoder (frozen).
        self.enc = AutoModel.from_pretrained(encoder_path)
        self.node_enc = AutoModel.from_pretrained(encoder_path)
        for p in self.node_enc.parameters():
            p.requires_grad = False
        # Optional partial freezing of content encoder's first N layers
        # (acts as regulariser for small datasets like Weibo's 5K samples).
        if freeze_bert_layers > 0:
            # Most HF backbones expose `encoder.layer.<idx>.*` and
            # `embeddings.*`; freeze embeddings + first N layers.
            for p in self.enc.embeddings.parameters():
                p.requires_grad = False
            for i, layer in enumerate(self.enc.encoder.layer):
                if i < freeze_bert_layers:
                    for p in layer.parameters():
                        p.requires_grad = False

        D = self.enc.config.hidden_size
        self.D = D
        # Optional projection if backbone dim mismatches.
        self.proj = nn.Identity()

        # CDG typed-edge attention stack.
        self.cdg_layers = nn.ModuleList([
            TypedEdgeAttention(D, heads=4, dropout=dropout)
            for _ in range(gnn_layers)
        ])
        self.cdg_norms = nn.ModuleList([nn.LayerNorm(D) for _ in range(gnn_layers)])

        # Reliability-gated fusion. Input = [h_text, h_cdg, delta, reliability(3)]
        gate_in_dim = D * 3 + 3
        if use_gate:
            self.gate = nn.Sequential(
                nn.Linear(gate_in_dim, D),
                nn.ReLU(),
                nn.Linear(D, 3),
                nn.Softmax(dim=-1),
            )
        else:
            # Equal weighting (1/3 each) when gate is disabled.
            self.register_buffer('gate_weights', torch.tensor([1 / 3, 1 / 3, 1 / 3]))

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(D, num_labels)

        # FTR auxiliary heads (v1-style hard + simple rationale supervision).
        if use_aux:
            aux_d = max(64, D // 2)
            # Hard rationale head: predict FTR_acc (binary correctness).
            self.td_hard_head = nn.Sequential(
                nn.Linear(D, aux_d), nn.ReLU(),
                nn.Linear(aux_d, 1), nn.Sigmoid(),
            )
            self.cs_hard_head = nn.Sequential(
                nn.Linear(D, aux_d), nn.ReLU(),
                nn.Linear(aux_d, 1), nn.Sigmoid(),
            )
            # Simple rationale head: predict FTR_pred (3-class real/fake/other).
            self.td_simple_head = nn.Sequential(
                nn.Linear(D, aux_d), nn.ReLU(),
                nn.Linear(aux_d, 3),
            )
            self.cs_simple_head = nn.Sequential(
                nn.Linear(D, aux_d), nn.ReLU(),
                nn.Linear(aux_d, 3),
            )

        # Loss modules.
        self.sccl = SCCLLoss(tau, w_hard)
        self.tamper = TamperMagnitudeLoss(tamper_m, tamper_eps)
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.bce = nn.BCELoss()
        self.ce3 = nn.CrossEntropyLoss()

    # ── helpers ──

    @torch.no_grad()
    def _encode_nodes(self, node_ids, node_mask):
        """[B, N, L] -> [B, N, D] using frozen node encoder."""
        B, N, L = node_ids.shape
        flat_ids = node_ids.view(B * N, L)
        flat_mask = node_mask.view(B * N, L)
        out = self.node_enc(input_ids=flat_ids, attention_mask=flat_mask)
        return out.last_hidden_state[:, 0, :].view(B, N, -1)

    def _encode_text(self, ids, mask):
        out = self.enc(input_ids=ids, attention_mask=mask)
        return out.last_hidden_state[:, 0, :]

    # ── forward ──

    def forward(self,
                content_ids, content_mask,
                twin_ids, twin_mask,
                style_ids, style_mask,
                node_ids, node_mask_seq,
                edge_index, edge_type, edge_weight,
                node_density, node_valid,
                reliability,
                labels: Optional[torch.Tensor] = None,
                has_twin: Optional[torch.Tensor] = None,
                has_style: Optional[torch.Tensor] = None,
                td_ids: Optional[torch.Tensor] = None,
                td_mask: Optional[torch.Tensor] = None,
                cs_ids: Optional[torch.Tensor] = None,
                cs_mask: Optional[torch.Tensor] = None,
                td_acc: Optional[torch.Tensor] = None,
                cs_acc: Optional[torch.Tensor] = None,
                td_pred: Optional[torch.Tensor] = None,
                cs_pred: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Args
        ----
        content_ids, content_mask:  [B, L]
        twin_ids, twin_mask:        [B, L] (padded if no twin)
        style_ids, style_mask:      [B, L] (padded if no style)
        node_ids, node_mask_seq:    [B, N, Ln]
        edge_index:                 [B, E, 2]  (-1 for padding)
        edge_type, edge_weight:     [B, E]
        node_density:               [B, N]
        node_valid:                 [B, N]  bool
        reliability:                [B, 3]  (rho, DM_norm, Gini)
        labels, has_twin, has_style: [B]
        td_ids, td_mask:            [B, L]   td_rationale (v1 FTR_2)
        cs_ids, cs_mask:            [B, L]   cs_rationale (v1 FTR_3)
        td_acc, cs_acc:             [B]      binary FTR correctness
        td_pred, cs_pred:           [B]      3-class FTR verdict
        """
        B = content_ids.size(0)
        device = content_ids.device

        # ── text branches ──
        h_news = self._encode_text(content_ids, content_mask)               # [B, D]
        h_twin = self._encode_text(twin_ids, twin_mask) if self.use_twin else h_news
        h_style = self._encode_text(style_ids, style_mask) if self.use_style else h_news

        # ── CDG branch ──
        if self.use_cdg:
            node_feat = self._encode_nodes(node_ids, node_mask_seq)         # [B, N, D]
            # Density-weighted: emphasise high-deviation nodes.
            weighted = node_feat * (1.0 + node_density[..., None])
            h = weighted
            for attn, norm in zip(self.cdg_layers, self.cdg_norms):
                h = norm(h + attn(h, edge_index, edge_type, edge_weight, node_valid))
            # Pool with density as soft attention weight.
            w = (node_density + 0.1) * node_valid.float()
            w_sum = w.sum(dim=1, keepdim=True) + 1e-8
            h_cdg = (h * w[..., None]).sum(dim=1) / w_sum                   # [B, D]
        else:
            h_cdg = torch.zeros_like(h_news)

        # ── rationale branch (direct FTR text via shared encoder) ──
        if td_ids is not None and cs_ids is not None:
            h_rat = 0.5 * (self._encode_text(td_ids, td_mask) +
                           self._encode_text(cs_ids, cs_mask))
        else:
            h_rat = h_news

        # ── tamper vector (kept for twin/style variants) ──
        if self.use_twin:
            delta = h_news - h_twin                                          # [B, D]
        else:
            delta = h_news - h_news                                          # zeros

        # ── gated fusion (text + CDG + rationale) ──
        gate_in = torch.cat([h_news, h_cdg, h_rat, reliability], dim=-1)     # [B, 3D+3]
        if self.use_gate:
            g = self.gate(gate_in)                                          # [B, 3]
        else:
            g = self.gate_weights.expand(B, -1)
        fused = g[:, 0:1] * h_news + g[:, 1:2] * h_cdg + g[:, 2:3] * h_rat   # [B, D]
        logits = self.classifier(self.dropout(fused))                        # [B, num_labels]

        out: Dict[str, Any] = {
            'logits': logits,
            'h_news': h_news,
            'h_cdg': h_cdg,
            'delta': delta,
            'gate': g,
            'reliability': reliability,
        }

        if labels is not None:
            loss_cls = self.ce(logits, labels)
            loss = loss_cls
            extras: Dict[str, float] = {'loss_cls': loss_cls.item()}

            if self.use_twin and has_twin is not None and has_twin.any():
                if self.alpha > 0:
                    l_sccl = self.sccl(h_news, h_twin, labels, has_twin)
                    loss = loss + self.alpha * l_sccl
                    extras['loss_sccl'] = float(l_sccl.item())
                if self.beta > 0:
                    l_tamper = self.tamper(h_news, h_twin, labels, has_twin)
                    loss = loss + self.beta * l_tamper
                    extras['loss_tamper'] = float(l_tamper.item())
            if self.use_style and has_style is not None and has_style.any() and self.gamma > 0:
                # Recompute style prediction through the same gate+classifier
                # but using h_style as the text branch.
                style_gate_in = torch.cat([h_style, h_cdg, delta, reliability], dim=-1)
                if self.use_gate:
                    g_s = self.gate(style_gate_in)
                else:
                    g_s = self.gate_weights.expand(B, -1)
                fused_s = g_s[:, 0:1] * h_style + g_s[:, 1:2] * h_cdg + g_s[:, 2:3] * delta
                logits_style = self.classifier(self.dropout(fused_s))
                l_style = style_invariance_loss(logits, logits_style, has_style)
                loss = loss + self.gamma * l_style
                extras['loss_style'] = float(l_style.item())

            # FTR auxiliary supervision (v1-style): encode td_rationale and
            # cs_rationale through the shared content encoder, then predict
            # their correctness (hard head) and verdict (simple head).
            if self.use_aux and self.alpha_aux > 0 and td_ids is not None and cs_ids is not None:
                h_td = self._encode_text(td_ids, td_mask)
                h_cs = self._encode_text(cs_ids, cs_mask)
                td_hard = self.td_hard_head(h_td).squeeze(-1)
                cs_hard = self.cs_hard_head(h_cs).squeeze(-1)
                td_simple = self.td_simple_head(h_td)
                cs_simple = self.cs_simple_head(h_cs)
                if td_acc is not None and cs_acc is not None:
                    l_hard = self.bce(td_hard, td_acc.clamp(0, 1)) + self.bce(cs_hard, cs_acc.clamp(0, 1))
                    loss = loss + self.alpha_aux * l_hard / 2.0
                    extras['loss_hard_aux'] = float(l_hard.item())
                if td_pred is not None and cs_pred is not None:
                    l_simple = self.ce3(td_simple, td_pred.long().clamp(0, 2)) + self.ce3(cs_simple, cs_pred.long().clamp(0, 2))
                    loss = loss + self.beta_aux * l_simple / 2.0
                    extras['loss_simple_aux'] = float(l_simple.item())

            out['loss'] = loss
            out['loss_extras'] = extras

        return out
