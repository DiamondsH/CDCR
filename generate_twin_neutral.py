#!/usr/bin/env python
"""Neutral twin generator: twin = content for ALL samples (no modification).

This avoids the data leak present in the heuristic generator (which added a
Chinese correction note only to fake samples). With neutral twins, the SCCL
becomes standard supervised contrastive learning:
  - For fake samples: anchor = content(twin), positive = other real, negative = other fake
  - For real samples: anchor = content(twin) = content(positive), no useful signal

The model still gets gradient from the contrastive loss for fake samples,
just without the self-reflective correction signal (which requires an LLM).
"""
from __future__ import annotations
import argparse
import json
import os


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
        for idx, item in enumerate(data):
            content = (item.get('content') or '').strip()
            rec = {
                'idx': idx,
                'id': item.get('id') or item.get('source_id') or idx,
                'twin': content,
                'len_ratio': 1.0,
                'passed_qc': bool(content),
                'src': 'neutral',
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + '\n')
        print(f'[neutral_twin] {len(data)} samples -> {args.out}')


if __name__ == '__main__':
    main()
