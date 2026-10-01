from __future__ import annotations

import re
from collections import Counter

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_TOKEN_RE = re.compile(r"\w+")
_STOPWORDS = frozenset(ENGLISH_STOP_WORDS)


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _lcs_length(a: list[str], b: list[str]) -> int:
    prev = [0] * (len(b) + 1)
    for token_a in a:
        curr = [0] * (len(b) + 1)
        for j, token_b in enumerate(b, start=1):
            if token_a == token_b:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[-1]


def rouge_l_score(prediction: str, reference: str) -> dict[str, float]:
    """Sentence-level ROUGE-L: LCS-based precision/recall/F1 over tokens."""
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)
    if not pred_tokens or not ref_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    lcs = _lcs_length(pred_tokens, ref_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def token_f1_score(prediction: str, reference: str) -> dict[str, float]:
    """QA-style token F1: multiset overlap of tokens between prediction and reference."""
    pred_tokens = _tokenize(prediction)
    ref_tokens = _tokenize(reference)
    if not pred_tokens or not ref_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    overlap = sum((Counter(pred_tokens) & Counter(ref_tokens)).values())
    if overlap == 0:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    precision = overlap / len(pred_tokens)
    recall = overlap / len(ref_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def _corpus_average(
    scores: list[dict[str, float]],
) -> dict[str, float]:
    n = len(scores)
    return {
        "precision": sum(s["precision"] for s in scores) / n,
        "recall": sum(s["recall"] for s in scores) / n,
        "f1": sum(s["f1"] for s in scores) / n,
    }


def rouge_l_corpus(predictions: list[str], references: list[str]) -> dict[str, float]:
    """Mean ROUGE-L precision/recall/F1 over a batch of (prediction, reference) pairs."""
    if len(predictions) != len(references):
        raise ValueError(
            f"predictions ({len(predictions)}) and references ({len(references)}) "
            "must have the same length"
        )
    scores = [rouge_l_score(p, r) for p, r in zip(predictions, references)]
    return _corpus_average(scores)


def token_f1_corpus(predictions: list[str], references: list[str]) -> dict[str, float]:
    """Mean token F1 precision/recall/F1 over a batch of (prediction, reference) pairs."""
    if len(predictions) != len(references):
        raise ValueError(
            f"predictions ({len(predictions)}) and references ({len(references)}) "
            "must have the same length"
        )
    scores = [token_f1_score(p, r) for p, r in zip(predictions, references)]
    return _corpus_average(scores)


def _content_tokenize(text: str) -> list[str]:
    return [t for t in _tokenize(text) if t not in _STOPWORDS]


def rouge_l_content_score(prediction: str, reference: str) -> dict[str, float]:
    """ROUGE-L computed on stopword-stripped token sequences.

    Plain ROUGE-L precision has a large context-free floor: an attacker that ignores the
    embedding and emits the corpus's most frequent tokens already scores ~0.37 on MS MARCO,
    because those tokens are stopwords that occur in most passages. Stripping stopwords removes
    that floor so the metric reflects information recovered about the passage's content.
    """
    pred_tokens = _content_tokenize(prediction)
    ref_tokens = _content_tokenize(reference)
    if not pred_tokens or not ref_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    lcs = _lcs_length(pred_tokens, ref_tokens)
    precision = lcs / len(pred_tokens)
    recall = lcs / len(ref_tokens)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def rouge_l_content_corpus(predictions: list[str], references: list[str]) -> dict[str, float]:
    """Mean content-word ROUGE-L precision/recall/F1 over (prediction, reference) pairs."""
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have the same length")
    return _corpus_average([rouge_l_content_score(p, r) for p, r in zip(predictions, references)])


def full_scores(predictions: list[str], references: list[str]) -> dict[str, float]:
    """Flat dict with standard and content-word ROUGE-L, for experiment CSV rows."""
    std = rouge_l_corpus(predictions, references)
    con = rouge_l_content_corpus(predictions, references)
    return {
        "rouge_l_precision": std["precision"], "rouge_l_recall": std["recall"], "rouge_l_f1": std["f1"],
        "content_precision": con["precision"], "content_recall": con["recall"], "content_f1": con["f1"],
    }
