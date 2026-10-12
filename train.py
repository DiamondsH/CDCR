#!/usr/bin/env python
"""CDCR training entry.

Mirrors the FACTGUARD training conventions (early stopping, val_f1 monitor,
TensorBoard, JSON dumps) but plugs in the CDCR model.

Required data:
  --data_dir/<train|val|test>.json         (source)
  --data_dir/nli_<split>.jsonl             (offline NLI results)
  --data_dir/twin_<split>.jsonl            (offline twin generation)
  --data_dir/style_<split>.jsonl           (offline style rewrites)
"""
from __future__ import annotations
import argparse
import json
import logging
import os
import random
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score, classification_report
from tqdm import tqdm

from models.cdcr import CDCR
from dataset import CDCRDataset, collate_v2

def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate(model, loader, device, threshold: float = 0.5):
    model.eval()
    all_labels, all_probs = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            out = model(
                content_ids=batch['content_ids'],
                content_mask=batch['content_mask'],
                twin_ids=batch['twin_ids'],
                twin_mask=batch['twin_mask'],
                style_ids=batch['style_ids'],
                style_mask=batch['style_mask'],
                node_ids=batch['node_ids'],
                node_mask_seq=batch['node_mask_seq'],
                edge_index=batch['edge_index'],
                edge_type=batch['edge_type'],
                edge_weight=batch['edge_weight'],
                node_density=batch['node_density'],
                node_valid=batch['node_mask'].bool(),
                reliability=batch['reliability'],
                td_ids=batch.get('td_ids'),
                td_mask=batch.get('td_mask'),
                cs_ids=batch.get('cs_ids'),
                cs_mask=batch.get('cs_mask'),
                labels=None,
            )
            logits = out['logits']
            probs = F.softmax(logits, dim=-1)[:, 1]
            all_probs.extend(probs.cpu().tolist())
            all_labels.extend(batch['label'].cpu().tolist())
    preds = (np.array(all_probs) >= threshold).astype(int)
    macro_f1 = f1_score(all_labels, preds, average='macro')
    acc = accuracy_score(all_labels, preds)
    return {
        'loss': 0.0,
        'f1': float(macro_f1),
        'acc': float(acc),
        'probs': all_probs,
        'preds': preds.tolist(),
        'labels': all_labels,
    }


def threshold_search(probs, labels, metric: str = 'f1'):
    """Find the threshold that maximises the given metric on the val set."""
    probs = np.array(probs)
    labels = np.array(labels)
    best_thr, best_score = 0.5, -1.0
    for thr in np.arange(0.20, 0.81, 0.005):
        preds = (probs >= thr).astype(int)
        if metric == 'f1':
            score = f1_score(labels, preds, average='macro')
        else:
            score = accuracy_score(labels, preds)
        if score > best_score:
            best_score = score
            best_thr = float(thr)
    return best_thr, best_score


def make_loader(split, args, tokenizer, shuffle):
    src = os.path.join(args.data_dir, f'{split}.json')
    nli = os.path.join(args.data_dir, f'nli_{split}.jsonl')
    twin = os.path.join(args.data_dir, f'twin_{split}{args.twin_suffix}.jsonl')
    style = os.path.join(args.data_dir, f'style_{split}.jsonl')
    ds = CDCRDataset(
        source_path=src,
        tokenizer=tokenizer,
        nli_path=nli if os.path.exists(nli) else None,
        twin_path=twin if os.path.exists(twin) else None,
        style_path=style if os.path.exists(style) else None,
        max_len=args.max_len,
        max_node_len=args.max_node_len,
        max_cdg_nodes=args.max_cdg_nodes,
        cdg_threshold_contra=args.cdg_threshold,
        cdg_threshold_entail=args.cdg_threshold_entail,
        use_nli_cdg=True,
        load_td_cs=not args.no_aux,
    )
    return DataLoader(
        ds, batch_size=args.batch_size,
        shuffle=shuffle, num_workers=args.num_workers,
        pin_memory=True, collate_fn=collate_v2,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--bert_path', required=True)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--max_len', type=int, default=170)
    parser.add_argument('--max_node_len', type=int, default=64)
    parser.add_argument('--max_cdg_nodes', type=int, default=14)
    parser.add_argument('--cdg_threshold', type=float, default=0.5)
    parser.add_argument('--cdg_threshold_entail', type=float, default=0.6)
    parser.add_argument('--early_stop', type=int, default=5)
    parser.add_argument('--gpu', type=str, default='0')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--alpha', type=float, default=0.3)
    parser.add_argument('--beta', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.05)
    parser.add_argument('--tau', type=float, default=0.07)
    parser.add_argument('--w_hard', type=float, default=2.0)
    parser.add_argument('--tamper_m', type=float, default=0.5)
    parser.add_argument('--tamper_eps', type=float, default=0.1)
    parser.add_argument('--no_twin', action='store_true')
    parser.add_argument('--no_style', action='store_true')
    parser.add_argument('--no_gate', action='store_true')
    parser.add_argument('--no_aux', action='store_true',
                        help='Disable FTR auxiliary supervision (v1-style).')
    parser.add_argument('--alpha_aux', type=float, default=0.4,
                        help='Weight of FTR hard-aux loss (BCE on FTR_acc).')
    parser.add_argument('--beta_aux', type=float, default=0.159,
                        help='Weight of FTR simple-aux loss (CE on FTR_pred).')
    parser.add_argument('--freeze_bert_layers', type=int, default=0,
                        help='Freeze first N BERT encoder layers (regulariser).')
    parser.add_argument('--no_cdg', action='store_true',
                        help='Disable CDG branch (pure text+rationale).')
    parser.add_argument('--class_weight', type=str, default='',
                        help='Optional comma-separated weights for CE loss, e.g. "1.2,0.8" (real,fake).')
    parser.add_argument('--twin_suffix', type=str, default='',
                        help='Optional suffix for twin jsonl files, e.g. "_neutral"')
    parser.add_argument('--model_name', type=str, default='CDCR')
    parser.add_argument('--data_name', type=str, default='cdcr')
    parser.add_argument('--save_log_dir', type=str, default='./logs')
    parser.add_argument('--save_param_dir', type=str, default='./param_model')
    parser.add_argument('--tensorboard_dir', type=str, default='./logs/tensorlog')
    parser.add_argument('--save_path', type=str, default=None)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    setup_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[V2] device={device} args={vars(args)}', flush=True)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.bert_path)

    # Directories.
    log_dir = os.path.join(args.save_log_dir, 'log')
    json_dir = os.path.join(args.save_log_dir, 'json',
                            f'{args.model_name}_{args.data_name}')
    param_log_dir = os.path.join(args.save_log_dir, 'param')
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)
    os.makedirs(param_log_dir, exist_ok=True)

    run_dir = os.path.join(args.save_param_dir, f'{args.model_name}_{args.data_name}', '1')
    os.makedirs(run_dir, exist_ok=True)
    save_path = args.save_path or os.path.join(run_dir, 'parameter_bert.pkl')

    # Logger.
    timestamp = datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
    param_log_path = os.path.join(param_log_dir,
                                  f'{args.model_name}_{args.data_name}_param.txt')
    file_handler = logging.FileHandler(param_log_path, mode='w', encoding='utf-8')
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    def log(msg):
        print(msg, flush=True)
        logger.info(msg)

    log(f'=== CDCR training ===')
    log(f'timestamp: {timestamp}')
    log(f'args: {vars(args)}')

    # Data.
    log('Loading data...')
    train_loader = make_loader('train', args, tokenizer, shuffle=True)
    val_loader = make_loader('val', args, tokenizer, shuffle=False)
    test_loader = make_loader('test', args, tokenizer, shuffle=False)
    log(f'Train={len(train_loader.dataset)} Val={len(val_loader.dataset)} '
        f'Test={len(test_loader.dataset)}')

    # Model.
    class_weights = None
    if args.class_weight:
        try:
            wr, wf = [float(x) for x in args.class_weight.split(',')]
            class_weights = torch.tensor([wr, wf], dtype=torch.float32)
            log(f'Using class weights (real,fake) = ({wr},{wf})')
        except Exception as e:
            log(f'Failed to parse class_weight, ignoring: {e}')
    model = CDCR(
        encoder_path=args.bert_path,
        alpha=args.alpha, beta=args.beta, gamma=args.gamma,
        alpha_aux=args.alpha_aux, beta_aux=args.beta_aux,
        tau=args.tau, w_hard=args.w_hard,
        tamper_m=args.tamper_m, tamper_eps=args.tamper_eps,
        use_twin=not args.no_twin,
        use_style=not args.no_style,
        use_gate=not args.no_gate,
        use_aux=not args.no_aux,
        freeze_bert_layers=args.freeze_bert_layers,
        class_weights=class_weights,
        use_cdg=not args.no_cdg,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f'Model trainable params: {n_params/1e6:.2f}M')

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=5e-5,
    )

    best_val_f1 = 0.0
    best_state = None
    no_improve = 0
    last_epoch = 0

    for epoch in range(args.epochs):
        last_epoch = epoch + 1
        model.train()
        total_loss = 0.0
        n_batches = 0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}')
        for batch in pbar:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            out = model(
                content_ids=batch['content_ids'],
                content_mask=batch['content_mask'],
                twin_ids=batch['twin_ids'],
                twin_mask=batch['twin_mask'],
                style_ids=batch['style_ids'],
                style_mask=batch['style_mask'],
                node_ids=batch['node_ids'],
                node_mask_seq=batch['node_mask_seq'],
                edge_index=batch['edge_index'],
                edge_type=batch['edge_type'],
                edge_weight=batch['edge_weight'],
                node_density=batch['node_density'],
                node_valid=batch['node_mask'].bool(),
                reliability=batch['reliability'],
                labels=batch['label'],
                has_twin=batch['has_twin'].bool(),
                has_style=batch['has_style'].bool(),
                td_ids=batch.get('td_ids'),
                td_mask=batch.get('td_mask'),
                cs_ids=batch.get('cs_ids'),
                cs_mask=batch.get('cs_mask'),
                td_acc=batch.get('td_acc'),
                cs_acc=batch.get('cs_acc'),
                td_pred=batch.get('td_pred'),
                cs_pred=batch.get('cs_pred'),
            )
            loss = out['loss']
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.item())
            n_batches += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_train_loss = total_loss / max(1, n_batches)
        val = evaluate(model, val_loader, device, threshold=0.5)
        log(f'Epoch {epoch+1}: train_loss={avg_train_loss:.4f}  '
            f'val_acc={val["acc"]:.4f}  val_f1={val["f1"]:.4f}')

        if val['f1'] > best_val_f1:
            best_val_f1 = val['f1']
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(model.state_dict(), save_path)
            log(f'  -> saved best (val_f1={best_val_f1:.4f})')
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= args.early_stop:
                log(f'Early stopping at epoch {epoch+1}')
                break

    log('=== Training done ===')

    # Restore best, then run threshold search on val + final test.
    if best_state is not None:
        model.load_state_dict(best_state)
    val = evaluate(model, val_loader, device, threshold=0.5)
    best_thr, best_val_thr_f1 = threshold_search(val['probs'], val['labels'], 'f1')
    log(f'Val best threshold = {best_thr:.3f} (f1={best_val_thr_f1:.4f})')

    test = evaluate(model, test_loader, device, threshold=best_thr)
    log(f'Test (thr={best_thr:.3f}): acc={test["acc"]:.4f}  f1={test["f1"]:.4f}')
    test_05 = evaluate(model, test_loader, device, threshold=0.5)
    log(f'Test (thr=0.5):   acc={test_05["acc"]:.4f}  f1={test_05["f1"]:.4f}')
    log('Classification report @ best threshold:')
    log(classification_report(test['labels'], test['preds'],
                              target_names=['real', 'fake']))

    # Persist results.
    summary = {
        'args': vars(args),
        'best_val_f1_0.5': best_val_f1,
        'best_threshold': best_thr,
        'val_f1_at_best_thr': best_val_thr_f1,
        'test_acc_at_best_thr': test['acc'],
        'test_f1_at_best_thr': test['f1'],
        'test_acc_0.5': test_05['acc'],
        'test_f1_0.5': test_05['f1'],
    }
    # Export probabilities in the unshuffled dataset order for paired analyses.
    import hashlib
    probability_dir = os.path.join(args.save_log_dir, 'probabilities')
    os.makedirs(probability_dir, exist_ok=True)
    probability_manifest = {'selection': 'validation_macro_f1', 'threshold': best_thr,
                            'label_names': ['real', 'fake'], 'splits': {}}
    for split, result in [('val', val), ('test', test)]:
        source_path = os.path.join(args.data_dir, split + '.json')
        with open(source_path, 'rb') as source_file:
            raw = source_file.read()
        rows = json.loads(raw)
        if len(rows) != len(result['labels']):
            raise ValueError('Dataset order export length mismatch: ' + split)
        ids = [str(row.get('source_id', row.get('id', row.get('idx', i))))
               for i, row in enumerate(rows)]
        filename = args.data_name + '_' + split + '.npz'
        np.savez_compressed(os.path.join(probability_dir, filename),
                            labels=np.asarray(result['labels']), probs=np.asarray(result['probs']),
                            sample_ids=np.asarray(ids), row_indices=np.arange(len(rows)))
        probability_manifest['splits'][split] = {
            'file': filename, 'source_sha256': hashlib.sha256(raw).hexdigest(),
            'count': len(rows), 'order': 'source_json_order'}
    with open(os.path.join(probability_dir, args.data_name + '_manifest.json'), 'w') as handle:
        json.dump(probability_manifest, handle, indent=2, ensure_ascii=False)
    summary['class_report_0.5'] = classification_report(
        test_05['labels'], test_05['preds'], target_names=['real', 'fake'],
        output_dict=True, zero_division=0)
    summary['class_report_val_threshold'] = classification_report(
        test['labels'], test['preds'], target_names=['real', 'fake'],
        output_dict=True, zero_division=0)
    with open(os.path.join(log_dir, f'{args.data_name}.json'), 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(os.path.join(json_dir, 'month_1.json'), 'w') as f:
        json.dump([summary], f, indent=2, ensure_ascii=False)
    log(f'Results saved to {log_dir} and {json_dir}')


if __name__ == '__main__':
    main()
