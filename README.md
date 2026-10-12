# CDCR: Causal Deviation and Counterfactual Reasoning

Core implementation and complete data splits for reviewer inspection.

## Contents

- `train.py`, `dataset.py`: training, validation-selected checkpoints, and input loading.
- `models/`: typed-edge graph attention, reliability-gated fusion, contrastive/style losses, graph construction, and NLI scoring. The model also retains rationale fusion and auxiliary rationale supervision used by this implementation.
- `compute_nli.py`, `generate_*.py`: optional preprocessing tools.
- `data/{zh,en}/`: source splits and aligned NLI, twin, and style inputs. JSONL `idx` is the zero-based row index in its source split; preserve row order.

| Dataset | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| Weibo21 (`zh`) | 5204 | 1951 | 1951 |
| GossipCop (`en`) | 3884 | 1274 | 1258 |

Source rows contain `content`, `label` (`real`/`fake`), `source_id`, verification/counterfactual rationale text (`td_rationale`/`cs_rationale`), and their categorical judgments (`td_pred`/`cs_pred`). Correctness targets for auxiliary training are derived from these judgments and labels when loading data. No evaluation accuracy fields are stored. NLI probabilities are preprocessing inputs, not detector predictions. `passed_qc` controls whether an augmentation is used.

**Input provenance:** supplied Chinese twins use the label-dependent heuristic in `generate_twin_heuristic.py`; supplied English twins use the unchanged-content fallback in `generate_twin_neutral.py`. They are not uniformly label-blind LLM-generated counterfactual corrections. `generate_twin.py` offers a separate LLM synthesis option. Supplied style rewrites have their acceptance masks retained; `generate_style.py` provides generation/basic screening and does not reproduce every additional offline screening step used for the supplied masks. Existing rationale text and prepared inputs permit training without regenerating them.

## Setup and training

Use Python 3.8 and install PyTorch for your CUDA environment, then:

```bash
pip install -r requirements.txt
```

Download `google-bert/bert-base-chinese` and `google-bert/bert-base-uncased` into `pretrained/bert-base-chinese` and `pretrained/bert-base-uncased`, respectively. Weights are not included. Run from this directory:

```bash
python train.py --data_dir data/zh --bert_path pretrained/bert-base-chinese \
  --epochs 15 --early_stop 4 --batch_size 16 --lr 1.5e-5 --seed 3759 \
  --freeze_bert_layers 9 --gpu 0
python train.py --data_dir data/en --bert_path pretrained/bert-base-uncased \
  --epochs 15 --early_stop 4 --batch_size 8 --lr 1.5e-5 --seed 3759 --gpu 0
```

These commands demonstrate the supplied implementation with available augmentations; they are not an ensemble or a reproduction command for every paper table. Training creates outputs locally when executed; no pre-existing outputs or model weights are distributed.

Optional NLI regeneration (repeat for each language/split):

```bash
python compute_nli.py --lang zh --data data/zh/train.json \
  --out data/zh/nli_train.jsonl --gpu 0
```

The default NLI models are `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` (Chinese) and `cross-encoder/nli-deberta-v3-large` (English); `--nli_model` accepts a local path. LLM generation tools accept `--model_path`; inspect `--help` for arguments. No weights or API credentials are bundled.

## Sources

News datasets are the Weibo21/GossipCop splits used in the ARG/FactGuard workflow. Respect the original dataset and model terms. Code adapted from the upstream FACTGUARD implementation retains its MIT notice in `LICENSE.third-party`; that notice covers the upstream code, not a new license for the datasets. This package contains only implementation and method inputs, without experiment results, run records, logs, or checkpoints.
