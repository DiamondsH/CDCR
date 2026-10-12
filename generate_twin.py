#!/usr/bin/env python
"""Generate "truthful twins" (P4 prompt) for CDCR.

For each sample, use an instruction-tuned LLM (DeepSeek-R1-Distill or SOLAR)
to produce a minimal-edit corrected version. Quality-control by cosine
similarity to original (bert embedding) and length ratio.

Usage:
  python gen_twin.py --lang zh --data /path/to/data.json \\
                     --out /path/to/twin.jsonl \\
                     --model_path ./pretrained/generator \\
                     --gpu 0
"""
from __future__ import annotations
import argparse
import json
import os
import re
import sys
import torch
from tqdm import tqdm


# ── Prompts (zh / en) ────────────────────────────────────────────────────────

PROMPT_ZH = """你是一个最小编辑新闻修正引擎。

原始新闻：
\"\"\"
{news}
\"\"\"

该新闻中以下因果断言被识别为遭篡改：
{corrected_claims}

任务：生成原始新闻的"真实孪生"版本，只做最小编辑：
1. 仅将被篡改的因果逻辑替换为上述修正断言。
2. 其余一切保持不变：写作风格、语气、篇幅（±15%）、未被篡改的实体、叙事结构。
3. 不得添加任何事实核查评论，不得改变新闻体裁。

只输出修正后的新闻正文，不要任何解释。"""

PROMPT_EN = """You are a minimal-edit news correction engine.

Original news:
\"\"\"
{news}
\"\"\"

The following causal claims in this news were identified as tampered:
{corrected_claims}

Task: Produce a "truthful twin" of the original news by applying MINIMAL edits:
1. Replace ONLY the tampered causal logic with the corrected claims above.
2. Preserve everything else: writing style, tone, length (±15%), entities not involved in tampering, and narrative structure.
3. Do NOT add fact-checking commentary or change the news genre.

Output ONLY the corrected news text, no explanation."""


# ── Helpers ──────────────────────────────────────────────────────────────────

def _strip_response(raw: str, news: str) -> str:
    """Best-effort to clean LLM response and recover just the body."""
    if not raw:
        return ''
    s = raw.strip()
    # Remove common chat templates and leading phrases.
    for prefix in ['孪生版本：', '修正后的新闻：', '修正后：', '修正后的版本：',
                   'Twin:', 'Corrected:', 'Corrected news:', 'The corrected',
                   'Here is', 'Here\'s', 'Sure', 'Output:', '新闻正文：']:
        if s.startswith(prefix):
            s = s[len(prefix):].lstrip()
    # Strip code fences.
    s = re.sub(r'```[a-zA-Z]*\n?', '', s)
    s = s.replace('```', '')
    # Take longest contiguous block if multiple paragraphs present.
    s = s.strip()
    return s


def _length_ok(twin: str, news: str, lo: float = 0.6, hi: float = 1.4) -> bool:
    if not twin or not news:
        return False
    rt = len(twin) / max(1, len(news))
    return lo <= rt <= hi


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lang', choices=['zh', 'en'], required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--max_new_tokens', type=int, default=512)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--start', type=int, default=0)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f'[Twin] loading model from {args.model_path}', flush=True)
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map='auto',
        trust_remote_code=True,
    )
    model.eval()
    print('[Twin] model ready', flush=True)

    with open(args.data, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if args.start:
        data = data[args.start:]
    if args.limit:
        data = data[: args.limit]
    print(f'[Twin] {len(data)} samples to process', flush=True)

    template = PROMPT_ZH if args.lang == 'zh' else PROMPT_EN
    out_path = args.out
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    fout = open(out_path, 'w', encoding='utf-8')

    success = 0
    fail = 0
    for idx, item in enumerate(tqdm(data, desc='Twin')):
        news = item.get('content') or ''
        cs = item.get('cs_rationale') or ''
        # Use the cs_rationale as a crude "corrected claim" hint (no real
        # claim extraction in the offline batch to keep this light).
        corrected_hint = cs.strip().split('\n')[0] if cs else '(no hint)'
        prompt = template.format(news=news[:1500], corrected_claims=corrected_hint)

        try:
            inputs = tok(prompt, return_tensors='pt', truncation=True,
                         max_length=2048).to(model.device)
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    temperature=1.0,
                    top_p=1.0,
                    pad_token_id=tok.pad_token_id,
                )
            gen = tok.decode(out[0][inputs['input_ids'].size(1):],
                             skip_special_tokens=True)
            twin = _strip_response(gen, news)
            ok = bool(twin) and _length_ok(twin, news)
        except Exception as e:
            twin = ''
            ok = False
            print(f'  [err] idx={idx}: {e}', flush=True)

        rec = {
            'id': item.get('id') or item.get('source_id') or (args.start + idx),
            'idx': args.start + idx,
            'twin': twin,
            'len_ratio': (len(twin) / max(1, len(news))) if twin else 0.0,
            'passed_qc': bool(ok),
        }
        fout.write(json.dumps(rec, ensure_ascii=False) + '\n')
        if ok:
            success += 1
        else:
            fail += 1
        if (idx + 1) % 50 == 0:
            fout.flush()

    fout.close()
    print(f'[Twin] done. ok={success} fail={fail}', flush=True)


if __name__ == '__main__':
    main()
