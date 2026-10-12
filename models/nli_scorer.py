"""NLI three-class scorer for CDCR.

Wraps either cross-encoder/nli-deberta-v3-large (English) or
MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7 (multilingual).

Returns softmax probabilities aligned to (contradiction, neutral, entailment)
order, regardless of the underlying model's id2label mapping.
"""
import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class NLIScorer:
    """Three-class NLI scorer.

    Args:
        model_name: HF repo id or local path.
        device: 'cuda' / 'cpu'.
        max_length: tokenization max length (default 256 is safe for NLI).
    """

    _TARGET = ('contradiction', 'neutral', 'entailment')

    def __init__(self, model_name: str, device: str = 'cuda', max_length: int = 256):
        self.model_name = model_name
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.max_length = max_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name).to(self.device).eval()
        self.id2label = self.model.config.id2label
        # Cache target index permutation once.
        self._perm = self._compute_perm()

    def _compute_perm(self) -> torch.Tensor:
        perm = []
        for tgt in self._TARGET:
            matched = None
            for j, lab in self.id2label.items():
                if str(lab).lower() == tgt:
                    matched = j
                    break
            if matched is None:
                # Fallback: assume the model's label order is the standard one.
                matched = self._TARGET.index(tgt)
            perm.append(int(matched))
        return torch.tensor(perm, dtype=torch.long)

    @torch.no_grad()
    def score(self, pairs, batch_size: int = 64):
        """Score NLI pairs.

        Args:
            pairs: list of (premise, hypothesis) tuples.
            batch_size: inference batch size.

        Returns:
            torch.Tensor of shape [N, 3] in (contra, neutral, entail) order.
        """
        out = []
        for i in range(0, len(pairs), batch_size):
            chunk = pairs[i:i + batch_size]
            premises = [p for p, _ in chunk]
            hyps = [h for _, h in chunk]
            enc = self.tokenizer(
                premises, hyps,
                truncation=True, max_length=self.max_length,
                padding=True, return_tensors='pt',
            ).to(self.device)
            logits = self.model(**enc).logits  # [b, num_labels]
            probs = F.softmax(logits, dim=-1)[:, self._perm.to(logits.device)]
            out.append(probs.cpu())
        if not out:
            return torch.zeros(0, 3)
        return torch.cat(out, dim=0)
