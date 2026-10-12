"""NLI-anchored Causal Deviation Graph (CDG) builder for CDCR.

Nodes: V-chain and C-chain reasoning steps (sentence-split), plus optional
       claim anchor nodes for explicit inter-claim causal edges.

Edges (4 types):
  0 = sequential      (within-chain: step_i <-> step_{i+1})
  1 = causal          (inter-claim dependency from depends_on)
  2 = contradiction   (cross-chain NLI: p_contra > tau_contra)
  3 = support         (cross-chain NLI: p_entail > tau_entail)

Returns:
  edge_index: LongTensor [2, E]
  edge_type:  LongTensor [E]
  edge_weight: FloatTensor [E]
  density:    FloatTensor [N]   (per-step contradiction density, normalised)
  DM:         float             (deviation mass = sum of d_p)
  Gini:       float             (Gini coefficient over {d_p})
  reliability: FloatTensor [3]  (rho, DM_norm, Gini)  sample-level scalar summary
"""
from __future__ import annotations
import re
from typing import List, Dict, Any, Optional
import math
import torch


# ── sentence splitting ───────────────────────────────────────────────────────

# Match end-of-sentence punctuation (EN + ZH), with optional closing
# quotes/brackets and optional trailing whitespace. This works for both
# English ("Hello. World!") and Chinese ("你好。世界！" — no space).
_SENT_RE = re.compile(r'(?<=[.!?。！？])\s*')


def split_steps(text: str, min_len: int = 10, max_steps: int = 8) -> List[str]:
    """Split text into reasoning steps. Both EN and ZH supported."""
    if not text or not text.strip():
        return []
    sents = _SENT_RE.split(text.strip())
    steps = [s.strip() for s in sents if len(s.strip()) >= min_len]
    if len(steps) > max_steps:
        # Keep first and last few to avoid blowing CDG node budget.
        head = steps[: max_steps - 2]
        tail = steps[-2:]
        steps = head + tail
    return steps


# ── Gini coefficient ─────────────────────────────────────────────────────────

def _gini(x: torch.Tensor, eps: float = 1e-8) -> float:
    x = x.clamp(min=0).flatten().to(torch.float64)
    n = x.numel()
    if n <= 1 or x.sum() < eps:
        return 0.0
    x_sorted, _ = torch.sort(x)
    idx = torch.arange(1, n + 1, dtype=torch.float64)
    g = (2.0 * (idx * x_sorted).sum() / (n * x_sorted.sum())) - (n + 1) / n
    return float(g.clamp(0, 1))


# ── CDG builder ──────────────────────────────────────────────────────────────

class CDGBuilderV2:
    def __init__(self,
                 nli_scorer=None,
                 tau_contra: float = 0.5,
                 tau_entail: float = 0.7,
                 use_nli: bool = True):
        """
        Args:
            nli_scorer: an NLIScorer instance; required if use_nli=True.
            tau_contra: threshold for contradiction edges.
            tau_entail: threshold for support edges.
            use_nli: if False, fall back to heuristic contradiction
                     (lexical negation + low overlap), useful for ablations.
        """
        self.nli = nli_scorer
        self.tau_contra = tau_contra
        self.tau_entail = tau_entail
        self.use_nli = use_nli

    def _heuristic_contradiction(self, a: str, b: str) -> float:
        """Cheap lexical heuristic used only when NLI is disabled."""
        wa = set(re.findall(r'\w+', a.lower()))
        wb = set(re.findall(r'\w+', b.lower()))
        if not wa or not wb:
            return 0.0
        overlap = len(wa & wb) / len(wa | wb)
        return float(max(0.0, 1.0 - overlap - 0.5))

    def build(self,
              v_text: str,
              c_text: str,
              claims: Optional[List[Dict[str, Any]]] = None,
              n_v_max: int = 6,
              n_c_max: int = 6) -> Dict[str, Any]:
        """Build a single CDG.

        Args:
            v_text: V-chain text (fact-verification rationale).
            c_text: C-chain text (counterfactual rationale).
            claims: list of dicts with optional 'depends_on' (int or None).
            n_v_max / n_c_max: per-chain step cap.
        """
        v_steps = split_steps(v_text, max_steps=n_v_max)
        c_steps = split_steps(c_text, max_steps=n_c_max)
        n_v, n_c = len(v_steps), len(c_steps)
        n_steps = n_v + n_c
        n_claims = max(len(claims or []), 0)
        N = n_steps + n_claims  # step nodes + claim anchor nodes
        anchor_ids = list(range(n_steps, N)) if n_claims > 0 else []

        src: List[int] = []
        dst: List[int] = []
        etype: List[int] = []
        eweight: List[float] = []
        density = torch.zeros(N)

        # ── 1) sequential edges within each chain ──
        for chain in (range(n_v), range(n_v, n_steps)):
            ids = list(chain)
            for a, b in zip(ids[:-1], ids[1:]):
                src += [a, b]
                dst += [b, a]
                etype += [0, 0]
                eweight += [1.0, 1.0]

        # ── 2) step -> claim-anchor edges (sequential) ──
        if n_claims > 0 and n_steps > 0:
            # Distribute steps evenly across claim anchors.
            per = max(1, n_steps // n_claims)
            for s in range(n_steps):
                j = min(s // per, n_claims - 1)
                aj = anchor_ids[j]
                src += [s, aj]
                dst += [aj, s]
                etype += [0, 0]
                eweight += [1.0, 1.0]

        # ── 3) inter-claim causal edges (depends_on) ──
        if claims:
            for j, cl in enumerate(claims):
                dep = cl.get('depends_on') if isinstance(cl, dict) else None
                if dep is None:
                    continue
                try:
                    dep_idx = int(dep) - 1
                except (TypeError, ValueError):
                    continue
                if 0 <= dep_idx < n_claims and dep_idx != j:
                    a_src = anchor_ids[dep_idx]
                    a_dst = anchor_ids[j]
                    src += [a_src, a_dst]
                    dst += [a_dst, a_src]
                    etype += [1, 1]
                    eweight += [1.0, 1.0]

        # ── 4) cross-chain NLI edges ──
        if n_v > 0 and n_c > 0:
            pairs = [(v, c) for v in v_steps for c in c_steps]
            if self.use_nli and self.nli is not None:
                probs = self.nli.score(pairs)  # [n_v*n_c, 3]
                k = 0
                for i in range(n_v):
                    for j in range(n_c):
                        p_contra, _, p_entail = probs[k].tolist()
                        k += 1
                        u, w = i, n_v + j
                        if p_contra > self.tau_contra:
                            src += [u, w]
                            dst += [w, u]
                            etype += [2, 2]
                            eweight += [p_contra, p_contra]
                            density[u] += p_contra
                            density[w] += p_contra
                        elif p_entail > self.tau_entail:
                            src += [u, w]
                            dst += [w, u]
                            etype += [3, 3]
                            eweight += [p_entail, p_entail]
            else:
                for i, v in enumerate(v_steps):
                    for j, c in enumerate(c_steps):
                        s = self._heuristic_contradiction(v, c)
                        if s > self.tau_contra:
                            u, w = i, n_v + j
                            src += [u, w]
                            dst += [w, u]
                            etype += [2, 2]
                            eweight += [s, s]
                            density[u] += s
                            density[w] += s

        # ── 5) density stats and tamper fingerprint ──
        step_density = density[:n_steps]
        DM = float(step_density.sum())
        G = _gini(step_density)
        max_d = step_density.max().item() if step_density.numel() else 0.0
        norm_density = step_density / max(max_d, 1e-8)

        # ── 6) empty-graph guard: single self-loop ──
        if not src:
            src, dst, etype, eweight = [0], [0], [0], [1.0]

        # ── 7) sample-level reliability scalars ──
        # rho: weighted NLI margin (mean over cross-chain pairs).
        if n_v > 0 and n_c > 0:
            if self.use_nli and self.nli is not None:
                # Recompute lightweight margin if NLI available; else skip.
                try:
                    probs = self.nli.score([(v, c) for v in v_steps for c in c_steps])
                    margin = float((probs[:, 2] - probs[:, 0]).mean().item())
                except Exception:
                    margin = 0.0
            else:
                margin = 0.0
        else:
            margin = 0.0
        # Normalise DM by log(1+N) so the value is in a reasonable range.
        DM_norm = DM / (math.log1p(n_steps) + 1e-8)
        reliability = torch.tensor([max(0.0, min(1.0, 0.5 + 0.5 * margin)),
                                    DM_norm, G], dtype=torch.float)

        return {
            'edge_index': torch.tensor([src, dst], dtype=torch.long),
            'edge_type': torch.tensor(etype, dtype=torch.long),
            'edge_weight': torch.tensor(eweight, dtype=torch.float),
            'density': norm_density,           # [n_steps]
            'n_v': n_v, 'n_c': n_c, 'n_steps': n_steps,
            'anchor_ids': anchor_ids,
            'DM': DM,
            'Gini': G,
            'reliability': reliability,        # [3] = (rho, DM_norm, Gini)
        }
