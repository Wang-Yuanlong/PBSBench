from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter


def normalize(text):
    text = unicodedata.normalize("NFKC", str(text)).casefold()
    return " ".join(re.sub(r"[^\w\s]", " ", text).split())


def _choice_text(prediction, options):
    value = str(prediction).strip()
    match = re.match(
        r"^\s*(?:(?:\(([A-Z])\)|\[([A-Z])\]|([A-Z])[.):])(?:\s|$)|([A-Z])\s*$)",
        value,
        re.IGNORECASE,
    )
    if match:
        label = next(group for group in match.groups() if group is not None)
        index = ord(label.upper()) - ord("A")
        if 0 <= index < len(options):
            return options[index]
    return value


def _bleu1(reference, prediction):
    reference, prediction = normalize(reference).split(), normalize(prediction).split()
    if not prediction:
        return 0.0
    overlap = sum((Counter(prediction) & Counter(reference)).values())
    precision = overlap / len(prediction)
    brevity = 1.0 if len(prediction) >= len(reference) else math.exp(
        1 - len(reference) / len(prediction)
    )
    return brevity * precision


def _rouge_l(reference, prediction):
    reference, prediction = normalize(reference).split(), normalize(prediction).split()
    if not reference or not prediction:
        return 0.0
    previous = [0] * (len(prediction) + 1)
    for ref_token in reference:
        current = [0]
        for index, pred_token in enumerate(prediction, 1):
            current.append(
                previous[index - 1] + 1
                if ref_token == pred_token
                else max(previous[index], current[-1])
            )
        previous = current
    lcs = previous[-1]
    precision, recall = lcs / len(prediction), lcs / len(reference)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _mean(values):
    return sum(values) / len(values) if values else None


def score_predictions(records, semantic_model=None):
    """Compute the deterministic metrics described in the PBSBench paper."""
    groups = {"true_false": [], "multiple_choice": [], "fill_blank": [], "open": []}
    for record in records:
        groups.setdefault(record["question_type"], []).append(record)

    true_false = [
        float(normalize(item["prediction"]).split()[:1] == normalize(item["answer"]).split()[:1])
        for item in groups["true_false"]
    ]
    multiple_choice = [
        float(
            normalize(_choice_text(item["prediction"], item.get("options", [])))
            == normalize(item["answer"])
        )
        for item in groups["multiple_choice"]
    ]
    fill_exact = [
        float(normalize(item["prediction"]) == normalize(item["answer"]))
        for item in groups["fill_blank"]
    ]
    fill_partial = [
        float(normalize(item["answer"]) in normalize(item["prediction"]))
        for item in groups["fill_blank"]
    ]
    open_bleu = [_bleu1(item["answer"], item["prediction"]) for item in groups["open"]]
    open_rouge = [_rouge_l(item["answer"], item["prediction"]) for item in groups["open"]]

    metrics = {
        "count": len(records),
        "true_false": {"count": len(true_false), "accuracy": _mean(true_false)},
        "multiple_choice": {
            "count": len(multiple_choice),
            "accuracy": _mean(multiple_choice),
        },
        "fill_blank": {
            "count": len(fill_exact),
            "exact_match": _mean(fill_exact),
            "partial_match": _mean(fill_partial),
        },
        "open": {
            "count": len(open_bleu),
            "bleu1": _mean(open_bleu),
            "rouge_l": _mean(open_rouge),
            "semantic_cosine": None,
        },
    }
    if semantic_model and groups["open"]:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(semantic_model)
        references = model.encode(
            [item["answer"] for item in groups["open"]], normalize_embeddings=True
        )
        predictions = model.encode(
            [item["prediction"] for item in groups["open"]], normalize_embeddings=True
        )
        metrics["open"]["semantic_cosine"] = float(
            (references * predictions).sum(axis=1).mean()
        )
    return metrics
