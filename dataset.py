"""CDCR dataset + collate.

Loads the source JSON and merges in offline-computed NLI / twin / style /
CDG JSONL. Builds a CDG per sample at construction time (cheap; CDG
construction uses already-computed NLI matrix or heuristic fallback).

Each sample returns:
  content_ids/mask       [L]            news article tokens
  twin_ids/mask          [L]            truthful-twin tokens
  style_ids/mask         [L]            style-rewritten tokens
  node_ids/node_mask_seq [N, Ln]        CDG step text tokens (one per step)
  edge_index             [E, 2]         padded graph edges (-1 = pad)
  edge_type              [E]            0=seq, 1=causal, 2=contra, 3=support
  edge_weight            [E]
  node_density           [N]            per-step contradiction density
  node_mask              [N]            1=valid, 0=pad
  reliability            [3]            (rho, DM_norm, Gini)
  label, has_twin, has_style             scalars
"""
from __future__ import annotations
import json
import math
import os
import re
from typing import Dict, List, Optional, Any
import torch
from torch.utils.data import Dataset


LABEL_DICT = {'real': 0, 'fake': 1, '0': 0, '1': 1, 0: 0, 1: 1}


def _map_label(label: Any) -> int:
    """Map label from string ('real'/'fake') or int to 0/1."""
    if isinstance(label, str):
        label = label.strip().lower()
    return int(LABEL_DICT.get(label, 0))


def _load_jsonl(path: str) -> Dict[Any, dict]:
    out = {}
    if not path or not os.path.exists(path):
        return out
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = rec.get('idx')
            if key is None:
                key = rec.get('id')
            if key is not None:
                out[key] = rec
    return out


def _gini(x):
    x = [max(0.0, float(v)) for v in x]
    s = sum(x)
    n = len(x)
    if n <= 1 or s < 1e-8:
        return 0.0
    xs = sorted(x)
    weighted = sum((i + 1) * v for i, v in enumerate(xs))
    g = (2.0 * weighted / (n * s)) - (n + 1) / n
    return float(max(0.0, min(1.0, g)))


def _split_steps(text: str, max_steps: int = 6) -> List[str]:
    if not text or not text.strip():
        return []
    # Match end-of-sentence punctuation (EN + ZH), allow 0+ trailing spaces
    # so that Chinese text (no space after 。！？) splits correctly.
    sents = re.split(r'(?<=[.!?。！？])\s*', text.strip())
    steps = [s.strip() for s in sents if len(s.strip()) >= 10]
    if len(steps) > max_steps:
        head = steps[: max_steps - 2]
        tail = steps[-2:]
        return head + tail
    return steps


class CDCRDataset(Dataset):
    """One row per source sample, augmented with offline NLI/twin/style."""

    def __init__(self,
                 source_path: str,
                 tokenizer,
                 nli_path: Optional[str] = None,
                 twin_path: Optional[str] = None,
                 style_path: Optional[str] = None,
                 max_len: int = 170,
                 max_node_len: int = 32,
                 max_cdg_nodes: int = 14,
                 cdg_threshold_contra: float = 0.5,
                 cdg_threshold_entail: float = 0.7,
                 use_nli_cdg: bool = True,
                 load_td_cs: bool = True):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.max_node_len = max_node_len
        self.max_cdg_nodes = max_cdg_nodes
        self.cdg_threshold_contra = cdg_threshold_contra
        self.cdg_threshold_entail = cdg_threshold_entail
        self.use_nli_cdg = use_nli_cdg
        self.load_td_cs = load_td_cs

        with open(source_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.data = data

        self.nli_map = _load_jsonl(nli_path) if nli_path else {}
        self.twin_map = _load_jsonl(twin_path) if twin_path else {}
        self.style_map = _load_jsonl(style_path) if style_path else {}

    def __len__(self):
        return len(self.data)

    def _tokenize(self, text: str, max_len: int):
        enc = self.tokenizer(
            text or '',
            max_length=max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        return enc['input_ids'].squeeze(0), enc['attention_mask'].squeeze(0)

    def _build_cdg(self, idx: int, td: str, cs: str):
        """Build CDG tensors and the list of node text strings."""
        nli = self.nli_map.get(idx)
        v_steps = _split_steps(td, max_steps=self.max_cdg_nodes // 2)
        c_steps = _split_steps(cs, max_steps=self.max_cdg_nodes // 2)
        v_steps = v_steps[: self.max_cdg_nodes // 2]
        c_steps = c_steps[: self.max_cdg_nodes // 2]
        n_v, n_c = len(v_steps), len(c_steps)
        node_texts = v_steps + c_steps
        n_steps = len(node_texts)
        N = max(n_steps, 1)

        src, dst, etype, eweight = [], [], [], []
        density = [0.0] * N

        for chain in (range(n_v), range(n_v, n_steps)):
            ids = list(chain)
            for a, b in zip(ids[:-1], ids[1:]):
                src += [a, b]; dst += [b, a]
                etype += [0, 0]; eweight += [1.0, 1.0]

        if n_v > 0 and n_c > 0 and nli is not None and nli.get('nli_matrix'):
            m = nli['nli_matrix']
            m_e = nli.get('entail_matrix') or [[0.0] * n_c for _ in range(n_v)]
            for i in range(min(n_v, len(m))):
                row = m[i]
                row_e = m_e[i] if i < len(m_e) else [0.0] * n_c
                for j in range(min(n_c, len(row))):
                    p = float(row[j])
                    p_e = float(row_e[j]) if j < len(row_e) else 0.0
                    if p > self.cdg_threshold_contra:
                        u, w = i, n_v + j
                        src += [u, w]; dst += [w, u]
                        etype += [2, 2]; eweight += [p, p]
                        density[u] += p
                        density[w] += p
                    elif p_e > self.cdg_threshold_entail:
                        # Support edge (entailment): V-step supports C-step
                        u, w = i, n_v + j
                        src += [u, w]; dst += [w, u]
                        etype += [3, 3]; eweight += [p_e, p_e]
        elif n_v > 0 and n_c > 0:
            for i, v in enumerate(v_steps):
                wa = set(re.findall(r'\w+', v.lower()))
                for j, c in enumerate(c_steps):
                    wb = set(re.findall(r'\w+', c.lower()))
                    if not wa or not wb:
                        continue
                    overlap = len(wa & wb) / len(wa | wb)
                    s = max(0.0, 1.0 - overlap - 0.4)
                    if s > self.cdg_threshold_contra:
                        u, w = i, n_v + j
                        src += [u, w]; dst += [w, u]
                        etype += [2, 2]; eweight += [s, s]
                        density[u] += s
                        density[w] += s

        if not src:
            src, dst, etype, eweight = [0], [0], [0], [1.0]
            if not node_texts:
                node_texts = ['[empty]']

        max_d = max(density) if density else 0.0
        norm_density = [d / max(max_d, 1e-8) for d in density]

        if nli is not None:
            margin = float(nli.get('nli_margin', 0.0))
            DM = float(nli.get('DM', sum(density)))
            G = float(nli.get('Gini', _gini(density)))
        else:
            margin = 0.0
            DM = sum(density)
            G = _gini(density)
        rho = max(0.0, min(1.0, 0.5 + 0.5 * margin))
        DM_norm = DM / (math.log1p(len(density)) + 1e-8)
        reliability = [rho, DM_norm, G]

        # Cap nodes by density.
        if len(node_texts) > self.max_cdg_nodes:
            order = sorted(range(len(density)),
                           key=lambda k: -density[k])[: self.max_cdg_nodes]
            keep = set(order)
            new_src, new_dst, new_type, new_weight = [], [], [], []
            for s, d, t, w in zip(src, dst, etype, eweight):
                if s in keep and d in keep:
                    new_src.append(s); new_dst.append(d)
                    new_type.append(t); new_weight.append(w)
            new_node_texts = [node_texts[k] for k in order]
            new_density = [density[k] for k in order]
            src, dst, etype, eweight = new_src, new_dst, new_type, new_weight
            node_texts = new_node_texts
            density = new_density
            N = len(node_texts)

        return {
            'node_texts': node_texts,
            'edge_src': src, 'edge_dst': dst,
            'edge_type': etype, 'edge_weight': eweight,
            'node_density': norm_density,
            'reliability': reliability,
            'N': N,
        }

    def __getitem__(self, idx: int):
        item = self.data[idx]
        td = item.get('td_rationale') or ''
        cs = item.get('cs_rationale') or ''
        content = item.get('content') or ''

        c_ids, c_mask = self._tokenize(content, self.max_len)

        # Tokenize td_rationale / cs_rationale for FTR auxiliary supervision.
        td_text = item.get('td_rationale') or ''
        cs_text = item.get('cs_rationale') or ''
        td_ids, td_mask = self._tokenize(td_text, self.max_len)
        cs_ids, cs_mask = self._tokenize(cs_text, self.max_len)
        # Map string labels to 0/1/2; default 0 (real) when missing.
        _pred_map = {'real': 0, 'fake': 1, 'other': 2}
        def _to_pred(v):
            if v is None:
                return 0
            if isinstance(v, int):
                return int(v)
            return _pred_map.get(str(v).lower(), 0)
        td_pred = _to_pred(item.get('td_pred'))
        cs_pred = _to_pred(item.get('cs_pred'))
        # Derive auxiliary correctness targets instead of storing evaluation fields.
        gold = _map_label(item['label'])
        td_acc = int(td_pred == gold)
        cs_acc = int(cs_pred == gold)

        twin_rec = self.twin_map.get(idx) or {}
        twin_text = twin_rec.get('twin') if twin_rec.get('passed_qc') else ''
        if not twin_text:
            twin_text = content
            has_twin = 0
        else:
            has_twin = 1
        t_ids, t_mask = self._tokenize(twin_text, self.max_len)

        style_rec = self.style_map.get(idx) or {}
        style_text = ''
        for key in ('formal', 'emotional', 'neutral'):
            v = (style_rec.get('variants') or {}).get(key) or {}
            if v.get('passed_qc') and v.get('text'):
                style_text = v['text']
                break
        if not style_text:
            style_text = content
            has_style = 0
        else:
            has_style = 1
        s_ids, s_mask = self._tokenize(style_text, self.max_len)

        cdg = self._build_cdg(idx, td, cs)
        node_texts = cdg['node_texts']
        N = cdg['N']
        src, dst, etype, eweight = cdg['edge_src'], cdg['edge_dst'], cdg['edge_type'], cdg['edge_weight']
        density = cdg['node_density']
        reliability = cdg['reliability']

        # Tokenize each node.
        node_ids_list, node_mask_list = [], []
        for txt in node_texts:
            nid, nm = self._tokenize(txt, self.max_node_len)
            node_ids_list.append(nid)
            node_mask_list.append(nm)
        # Pad nodes.
        while len(node_ids_list) < self.max_cdg_nodes:
            pad_ids = torch.zeros(self.max_node_len, dtype=torch.long)
            pad_mask = torch.zeros(self.max_node_len, dtype=torch.long)
            node_ids_list.append(pad_ids)
            node_mask_list.append(pad_mask)
            density.append(0.0)
        if len(node_ids_list) > self.max_cdg_nodes:
            node_ids_list = node_ids_list[: self.max_cdg_nodes]
            node_mask_list = node_mask_list[: self.max_cdg_nodes]
            density = density[: self.max_cdg_nodes]
        node_ids = torch.stack(node_ids_list)            # [N, Ln]
        node_mask_seq = torch.stack(node_mask_list)      # [N, Ln]
        node_mask = torch.tensor(
            [1.0] * min(N, self.max_cdg_nodes) +
            [0.0] * max(0, self.max_cdg_nodes - N),
            dtype=torch.float,
        )

        # Edges: pad to N*N; -1 for padding.
        E = self.max_cdg_nodes * self.max_cdg_nodes
        edge_index = [[s, d] for s, d in zip(src, dst)]
        if len(edge_index) > E:
            edge_index = edge_index[:E]
            eweight = eweight[:E]
            etype = etype[:E]
        while len(edge_index) < E:
            edge_index.append([-1, -1])
            eweight.append(0.0)
            etype.append(-1)
        edge_index = torch.tensor(edge_index, dtype=torch.long)
        edge_type = torch.tensor(etype, dtype=torch.long)
        edge_weight = torch.tensor(eweight, dtype=torch.float)
        node_density = torch.tensor(density, dtype=torch.float)
        reliability = torch.tensor(reliability, dtype=torch.float)

        label = _map_label(item.get('label', 0))
        out = {
            'content_ids': c_ids,
            'content_mask': c_mask,
            'twin_ids': t_ids,
            'twin_mask': t_mask,
            'style_ids': s_ids,
            'style_mask': s_mask,
            'node_ids': node_ids,
            'node_mask_seq': node_mask_seq,
            'edge_index': edge_index,
            'edge_type': edge_type,
            'edge_weight': edge_weight,
            'node_density': node_density,
            'node_mask': node_mask,
            'reliability': reliability,
            'label': torch.tensor(label, dtype=torch.long),
            'has_twin': torch.tensor(has_twin, dtype=torch.float),
            'has_style': torch.tensor(has_style, dtype=torch.float),
        }
        if self.load_td_cs:
            out['td_ids'] = td_ids
            out['td_mask'] = td_mask
            out['cs_ids'] = cs_ids
            out['cs_mask'] = cs_mask
            out['td_acc'] = torch.tensor(td_acc, dtype=torch.float)
            out['cs_acc'] = torch.tensor(cs_acc, dtype=torch.float)
            out['td_pred'] = torch.tensor(td_pred, dtype=torch.long)
            out['cs_pred'] = torch.tensor(cs_pred, dtype=torch.long)
        return out


def collate_v2(batch):
    out = {}
    for k in batch[0].keys():
        out[k] = torch.stack([b[k] for b in batch], dim=0)
    return out
