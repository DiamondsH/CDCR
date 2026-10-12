#!/usr/bin/env python
"""Offline NLI scoring for CDCR.

For each sample with td_rationale and cs_rationale:
  1. Split each rationale into steps (sentence-level)
  2. Run NLI on all (V-step, C-step) pairs
  3. Compute (kappa, nli_margin, density, DM, Gini) per sample
  4. Write to JSONL keyed by sample id (use index in the source list)

Usage:
  python compute_nli.py --lang zh --data /path/to/data.json \\
                        --out /path/to/nli.jsonl --gpu 0
"""
from __future__ import annotations
import argparse
import json
import math
import os
import sys
from typing import List, Dict, Any

import torch
from tqdm import tqdm

from models.nli_scorer import NLIScorer
from models.cdg_builder import CDGBuilderV2, split_steps

def _gini(x):
    x = [max(0.0, float(v)) for v in x]
    s = sum(x)
    n = len(x)
    if n <= 1 or s < 1e-8:
        return 0.0
    xs = sorted(x)
    cum = 0.0
    weighted = 0.0
    for i, v in enumerate(xs, start=1):
        weighted += i * v
    g = (2.0 * weighted / (n * s)) - (n + 1) / n
    return float(max(0.0, min(1.0, g)))


def score_split(rationale: str, max_steps: int = 6) -> List[str]:
    return split_steps(rationale, max_steps=max_steps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang', choices=['zh', 'en'], required=True)
    parser.add_argument('--data', required=True, help='Path to source JSON')
    parser.add_argument('--out', required=True, help='Output JSONL path')
    parser.add_argument('--nli_model', default=None,
                        help='Override NLI model (otherwise chosen by --lang)')
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--max_steps', type=int, default=6)
    parser.add_argument('--max_pairs', type=int, default=64,
                        help='Cap cross-chain pairs per sample for speed.')
    parser.add_argument('--limit', type=int, default=0,
                        help='Process only first N samples (0 = all).')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.nli_model:
        nli_name = args.nli_model
    elif args.lang == 'zh':
        nli_name = 'MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7'
    else:
        nli_name = 'cross-encoder/nli-deberta-v3-large'

    print(f'[NLI] lang={args.lang} model={nli_name} device={device}', flush=True)
    nli = NLIScorer(nli_name, device=device, max_length=256)

    with open(args.data, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if args.limit:
        data = data[: args.limit]
    print(f'[NLI] {len(data)} samples to process', flush=True)

    out_path = args.out
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)

    with open(out_path, 'w', encoding='utf-8') as fout:
        for idx, item in enumerate(tqdm(data, desc='NLI')):
            td = item.get('td_rationale') or ''
            cs = item.get('cs_rationale') or ''
            v_steps = score_split(td, max_steps=args.max_steps)
            c_steps = score_split(cs, max_steps=args.max_steps)
            n_v, n_c = len(v_steps), len(c_steps)
            sample_id = item.get('id') or item.get('source_id') or idx
            record = {
                'id': sample_id,
                'idx': idx,
                'label': item.get('label'),
                'n_v': n_v,
                'n_c': n_c,
                'v_steps': v_steps,
                'c_steps': c_steps,
            }

            if n_v == 0 or n_c == 0:
                record.update({
                    'nli_matrix': [],
                    'kappa': 0.0,
                    'nli_margin': 0.0,
                    'density': [],
                    'DM': 0.0,
                    'Gini': 0.0,
                })
                fout.write(json.dumps(record, ensure_ascii=False) + '\n')
                continue

            pairs = [(v, c) for v in v_steps for c in c_steps]
            if len(pairs) > args.max_pairs:
                # Sub-sample to control cost; deterministic.
                step = max(1, len(pairs) // args.max_pairs)
                pairs = pairs[::step][: args.max_pairs]
            probs = nli.score(pairs)  # [P, 3] in (contra, neutral, entail)

            # Recover full n_v*n_c matrices (for samples with > max_pairs we keep
            # the sub-sampled view; for smaller samples it's full).
            # contra_matrix: p_contra  (used for contradiction edges)
            # entail_matrix: p_entail (used for support edges)
            contra_matrix = []
            entail_matrix = []
            k = 0
            for i in range(n_v):
                row_c = []
                row_e = []
                for j in range(n_c):
                    if k < len(probs):
                        row_c.append(float(probs[k, 0]))
                        row_e.append(float(probs[k, 2]))
                        k += 1
                    else:
                        row_c.append(0.0)
                        row_e.append(0.0)
                contra_matrix.append(row_c)
                entail_matrix.append(row_e)

            # Per-step density: sum over cross-chain edges where this step is
            # the tail.
            step_density = [0.0] * (n_v + n_c)
            for i in range(min(n_v, len(probs) // max(1, n_c))):
                for j in range(n_c):
                    p_idx = i * n_c + j
                    if p_idx < len(probs):
                        c_p = float(probs[p_idx, 0])
                        step_density[i] += c_p
                        step_density[n_v + j] += c_p

            DM = sum(step_density)
            G = _gini(step_density)

            # Reliability margin = mean(p_entail - p_contra).
            margin = float((probs[:, 2] - probs[:, 0]).mean().item())

            # Kappa proxy: fraction of edges where p_contra > 0.5
            n_contra = int((probs[:, 0] > 0.5).sum().item())
            kappa = n_contra / max(1, len(probs))

            record.update({
                'nli_matrix': contra_matrix,
                'entail_matrix': entail_matrix,
                'kappa': float(kappa),
                'nli_margin': float(margin),
                'density': step_density,
                'DM': float(DM),
                'Gini': float(G),
            })
            fout.write(json.dumps(record, ensure_ascii=False) + '\n')

    print(f'[NLI] done -> {out_path}', flush=True)


if __name__ == '__main__':
    main()
