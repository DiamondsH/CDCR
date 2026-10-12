#!/usr/bin/env python
"""Heuristic twin generator: use cs_rationale as a "truthful twin".

For each news sample, produce a twin text by combining the original content
with a corrected causal statement derived from cs_rationale. This is a
lightweight approximation of LLM-generated twins when no suitable
instruction-tuned LLM is available in the environment.

Output: JSONL with {idx, twin, passed_qc, len_ratio}.
"""
from __future__ import annotations
import argparse
import json
import os
from typing import List


def heuristic_twin(content: str, cs: str, label: int) -> str:
    """Combine content with a counterfactual correction note.

    For real (label=0): keep content as-is (no edit needed).
    For fake (label=1): append a brief correction note derived from cs.
    """
    content = (content or '').strip()
    cs = (cs or '').strip()
    if not content:
        return ''
    if label == 0 or not cs:
        return content
    # Use first sentence of cs as the correction.
    first = cs.split('。')[0].split('.')[0].strip()
    if not first:
        first = cs[:60]
    note = f'（事实核查：根据权威信息，{first}。）'
    return content + note


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()

    with open(args.data, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if args.limit:
        data = data[: args.limit]

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as fout:
        n_ok = 0
        for idx, item in enumerate(data):
            content = item.get('content') or ''
            cs = item.get('cs_rationale') or ''
            label = int(item.get('label', 0))
            twin = heuristic_twin(content, cs, label)
            ratio = (len(twin) / max(1, len(content))) if twin else 0.0
            ok = 0.6 <= ratio <= 1.4 and len(twin) >= 20
            if ok:
                n_ok += 1
            rec = {
                'idx': idx,
                'id': item.get('id') or item.get('source_id') or idx,
                'twin': twin,
                'len_ratio': ratio,
                'passed_qc': bool(ok),
                'src': 'heuristic_cs',
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + '\n')
        print(f'[heuristic_twin] {len(data)} samples, {n_ok} passed QC -> {args.out}')


if __name__ == '__main__':
    main()
