"""Loss functions for CDCR.

- SCCLLoss: self-reflective hard-negative contrastive learning.
    anchor = h(n*), positive = batch real news, hard negative = h(n) (itself)
- TamperMagnitudeLoss: enforces ||delta|| large for fake, small for real.
- style_invariance_loss: bidirectional KL between p(n) and p(n_s).
- CausalDistillLoss: 3-level distillation (L1 logits / L2 detampered / L3 attention).
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Self-reflective Contrastive Loss ────────────────────────────────────────

class SCCLLoss(nn.Module):
    """Self-reflective hard-negative contrastive loss.

    For each fake sample i in batch (y_i = 1, has_twin_i = True):
        anchor   = h(n_i*)                  (twin encoding)
        positive = h(n_j) for j in real set
        hard neg = h(n_i)                   (the tampered original)
        other neg= h(n_k) for k in fake set, k != i

    Loss is InfoNCE-style over positive set vs all (weighted) negatives.
    """

    def __init__(self, temperature: float = 0.07, w_hard: float = 2.0):
        super().__init__()
        self.tau = temperature
        self.w_hard = w_hard

    def forward(self, h_news: torch.Tensor, h_twin: torch.Tensor,
                labels: torch.Tensor, has_twin: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h_news:  [B, D] raw news hidden states (will be L2-normalised).
            h_twin:  [B, D] twin hidden states (will be L2-normalised).
            labels:  [B]    0=real, 1=fake.
            has_twin:[B] bool.

        Returns:
            scalar loss (InfoNCE-style, always non-negative).
        """
        h_news = F.normalize(h_news, dim=-1)
        h_twin = F.normalize(h_twin, dim=-1)

        fake_mask = (labels == 1) & has_twin
        real_mask = (labels == 0) & has_twin
        fake_idx = fake_mask.nonzero(as_tuple=True)[0]
        real_idx = real_mask.nonzero(as_tuple=True)[0]
        if fake_idx.numel() == 0 or real_idx.numel() == 0:
            return h_news.new_zeros(())

        anchors = h_twin[fake_idx]                    # [F, D]
        pos = h_news[real_idx]                        # [R, D]
        hard_neg = h_news[fake_idx]                   # [F, D]

        # Standard InfoNCE: denom includes the positive term as well.
        sim_with_pos = anchors @ pos.T                          # [F, R]
        sim_with_all = anchors @ h_news.T                       # [F, B]

        pos_logsumexp = torch.logsumexp(sim_with_pos / self.tau, dim=-1)   # [F]
        all_logsumexp = torch.logsumexp(sim_with_all / self.tau, dim=-1)   # [F]

        # Hard-negative bias: down-weight the diagonal-like fake term.
        hard_sim = (anchors * hard_neg).sum(dim=-1) / self.tau   # [F]
        # log((exp(all) + (w_hard-1)*exp(hard)) / exp(pos)) =
        #     logsumexp(all, hard with weight) - pos_logsumexp
        # Since we want the loss to penalise the model for putting the fake
        # pair close together, add an extra (w_hard - 1) * exp(hard) to the
        # denominator by including it in the logsumexp via concatenation.
        if self.w_hard > 1.0:
            # Concat [sim_with_all, hard_sim with weight] and take logsumexp.
            F_n = anchors.size(0)
            weighted_hard = hard_sim + torch.log(
                torch.tensor(self.w_hard - 1.0, device=hard_sim.device,
                             dtype=hard_sim.dtype))
            stacked = torch.cat([sim_with_all / self.tau,
                                 weighted_hard.unsqueeze(-1)], dim=-1)  # [F, B+1]
            denom_logsumexp = torch.logsumexp(stacked, dim=-1)          # [F]
        else:
            denom_logsumexp = all_logsumexp

        loss = -(pos_logsumexp - denom_logsumexp)
        # Clamp to non-negative (numerical safety).
        loss = torch.clamp(loss, min=0.0)
        return loss.mean()


# ── Tamper magnitude regulariser ────────────────────────────────────────────

class TamperMagnitudeLoss(nn.Module):
    """Push ||delta|| away from real (small) and toward large for fake.

    L = mean( relu(m - ||delta||_fake) ) + mean( relu(||delta||_real - eps) )
    """

    def __init__(self, margin: float = 0.5, eps: float = 0.1):
        super().__init__()
        self.m = margin
        self.eps = eps

    def forward(self, h_news: torch.Tensor, h_twin: torch.Tensor,
                labels: torch.Tensor, has_twin: torch.Tensor) -> torch.Tensor:
        if not has_twin.any():
            return h_news.new_zeros(())
        delta = (h_news - h_twin).norm(dim=-1)  # [B]
        fake = (labels == 1) & has_twin
        real = (labels == 0) & has_twin
        loss = h_news.new_zeros(())
        if fake.any():
            loss = loss + F.relu(self.m - delta[fake]).mean()
        if real.any():
            loss = loss + F.relu(delta[real] - self.eps).mean()
        return loss


# ── Style invariance loss ───────────────────────────────────────────────────

def style_invariance_loss(logits_orig: torch.Tensor,
                          logits_style: torch.Tensor,
                          mask: torch.Tensor) -> torch.Tensor:
    """Bidirectional KL between p(n) and p(n_s) over masked samples.

    Used to regularise the model to be robust to surface-level style edits.
    """
    if not mask.any():
        return logits_orig.new_zeros(())
    # Use softmax (not log_softmax) + manual log for stable, version-clean graph.
    p = F.softmax(logits_orig[mask], dim=-1)
    q = F.softmax(logits_style[mask], dim=-1)
    # KL(p || q) = sum p * (log p - log q); add small eps to avoid log(0).
    eps = 1e-8
    kl_pq = (p * (p.clamp_min(eps).log() - q.clamp_min(eps).log())).sum(dim=-1).mean()
    kl_qp = (q * (q.clamp_min(eps).log() - p.clamp_min(eps).log())).sum(dim=-1).mean()
    return 0.5 * (kl_pq + kl_qp)


# ── 3-level Causal distillation loss ────────────────────────────────────────

class CausalDistillLoss(nn.Module):
    """L1 logits-KD + L2 detampered representation + L3 deviation attention.

    Returns (loss, (l1, l2, l3)) for logging.
    """

    def __init__(self, T: float = 2.0):
        super().__init__()
        self.T = T

    def forward(self,
                s_logits: torch.Tensor,
                t_logits: torch.Tensor,
                s_hidden: torch.Tensor,
                t_hidden_detampered: torch.Tensor,
                s_attn: torch.Tensor,
                t_deviation: torch.Tensor,
                lam: tuple = (1.0, 1.0, 0.5)):
        l1 = F.kl_div(
            F.log_softmax(s_logits / self.T, dim=-1),
            F.softmax(t_logits.detach() / self.T, dim=-1),
            reduction='batchmean',
        ) * self.T ** 2
        l2 = F.mse_loss(s_hidden, t_hidden_detampered.detach())
        l3 = F.kl_div(
            F.log_softmax(s_attn, dim=-1),
            F.softmax(t_deviation.detach(), dim=-1),
            reduction='batchmean',
        )
        total = lam[0] * l1 + lam[1] * l2 + lam[2] * l3
        return total, (l1.item(), l2.item(), l3.item())
