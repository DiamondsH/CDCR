#!/usr/bin/env python
"""Generate style variants (P5 prompt) for CDCR.

For each sample, produce 3 rewrites in (formal, emotional, neutral) styles.
Used to train L_style invariance regulariser.

Usage:
  python gen_style.py --lang zh --data /path/to/data.json \\
                      --out /path/to/style.jsonl \\
                      --model_path ./pretrained/generator
"""
from __future__ import annotations
import argparse
import json
import os
import re
import sys
import torch
from tqdm import tqdm


PROMPT_ZH = """请将以下新闻改写为{style}风格。
风格选项：[严肃权威的通讯社风格 / 高度情绪化、耸动的自媒体风格 / 平实中立的复述风格]
关键要求：所有事实、事件、实体与因果断言必须完全保留，只改变写作风格。
只输出改写后的文本。

新闻：
\"\"\"
{news}
\"\"\""""

PROMPT_EN = """Rewrite the following news in a {style} style.
Styles: [formal-authoritative news agency / highly emotional and sensational / plain neutral paraphrase]
CRITICAL: Preserve ALL facts, events, entities, and causal claims exactly. Change ONLY the writing style.
Output only the rewritten text.

News:
\"\"\"
{news}
\"\"\""""

STYLE_ZH = {
    'formal': '严肃权威的通讯社风格',
    'emotional': '高度情绪化、耸动的自媒体风格',
    'neutral': '平实中立的复述风格',
}
STYLE_EN = {
    'formal': 'formal-authoritative news agency',
    'emotional': 'highly emotional and sensational',
    'neutral': 'plain neutral paraphrase',
}


def _strip(raw: str) -> str:
    if not raw:
        return ''
    s = raw.strip()
    s = re.sub(r'```[a-zA-Z]*\n?', '', s)
    s = s.replace('```', '')
    return s.strip()


def _length_ok(t: str, n: str, lo: float = 0.7, hi: float = 1.3) -> bool:
    if not t or not n:
        return False
    return lo <= (len(t) / max(1, len(n))) <= hi


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
    parser.add_argument('--styles', nargs='+', default=['formal', 'emotional', 'neutral'])
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f'[Style] loading model from {args.model_path}', flush=True)
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
    print('[Style] model ready', flush=True)

    with open(args.data, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if args.start:
        data = data[args.start:]
    if args.limit:
        data = data[: args.limit]
    print(f'[Style] {len(data)} samples, styles={args.styles}', flush=True)

    template = PROMPT_ZH if args.lang == 'zh' else PROMPT_EN
    style_map = STYLE_ZH if args.lang == 'zh' else STYLE_EN

    out_path = args.out
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    fout = open(out_path, 'w', encoding='utf-8')

    n_ok = 0
    n_total = 0
    for idx, item in enumerate(tqdm(data, desc='Style')):
        news = item.get('content') or ''
        rec = {
            'id': item.get('id') or item.get('source_id') or (args.start + idx),
            'idx': args.start + idx,
            'variants': {},
        }
        for style_key in args.styles:
            n_total += 1
            label = style_map[style_key]
            prompt = template.format(style=label, news=news[:1500])
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
                txt = _strip(gen)
                ok = _length_ok(txt, news)
                rec['variants'][style_key] = {
                    'text': txt,
                    'len_ratio': (len(txt) / max(1, len(news))) if txt else 0.0,
                    'passed_qc': bool(ok),
                }
                if ok:
                    n_ok += 1
            except Exception as e:
                rec['variants'][style_key] = {
                    'text': '', 'len_ratio': 0.0, 'passed_qc': False,
                }
                print(f'  [err] idx={idx} style={style_key}: {e}', flush=True)
        fout.write(json.dumps(rec, ensure_ascii=False) + '\n')
        if (idx + 1) % 50 == 0:
            fout.flush()

    fout.close()
    print(f'[Style] done. ok={n_ok}/{n_total}', flush=True)


if __name__ == '__main__':
    main()
