"""Evaluation-only rare-group matching and finite independent prediction reference."""

from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score

from super_deli.benchmarking.bootstrap_split_audit import logdensity
from super_deli.io.config import read_json


def recovery(truth, labels):
    truth, labels = np.asarray(truth), np.asarray(labels)
    groups = [set(np.flatnonzero(truth == k)) for k in np.unique(truth)]
    smallest = min(range(len(groups)), key=lambda k: (len(groups[k]), k))
    group = groups[smallest]
    parts = [set(np.flatnonzero(labels == k)) for k in np.unique(labels)]
    best = max(range(len(parts)), key=lambda i: (len(group & parts[i]) / len(group | parts[i]), -i))
    part = parts[best]
    hit = len(group & part)
    jaccard = hit / len(group | part)
    ari = float(adjusted_rand_score(truth, labels))
    k = len(parts)
    return dict(
        selected_k=k,
        generating_k=len(groups),
        ari=ari,
        rare_size=len(group),
        rare_jaccard=jaccard,
        rare_precision=hit / len(part),
        rare_recall=hit / len(group),
        correct=k == len(groups) and ari >= 0.8 and jaccard >= 0.8,
    )


def external_reference(subsample, report):
    subsample, report = Path(subsample), Path(report)
    events = read_json(subsample / "observed-pipeline/events.json")
    predictions = iter(read_json(subsample / "predictions.json"))
    scores = read_json(report / "predictive_replicates.json")
    values = []
    checked = 0
    rows = []
    for event, record in zip(events, scores, strict=True):
        if event["status"] != "complete":
            rows.append(dict(status="unavailable_native"))
            continue
        prediction = next(predictions)
        if record["status"] != "complete":
            rows.append(dict(status="unavailable_predictive"))
            continue
        try:
            with np.load(subsample / prediction["arrays"], allow_pickle=False) as a:
                with np.load(report / "scores" / record["arrays"], allow_pickle=False) as b:
                    base, _, _ = logdensity(a["evaluation_scores"], b)
                np.testing.assert_allclose(
                    base, a["k1_evaluation_logdensity"], rtol=1e-7, atol=1e-9
                )
                chosen = a[f"k{prediction['selected_k']}_evaluation_logdensity"]
                gain = chosen - base
                if prediction["selected_k"] == 1:
                    np.testing.assert_allclose(gain, 0, rtol=1e-7, atol=1e-9)
                    gain = np.zeros_like(gain)
                if not np.isfinite(gain).all():
                    raise FloatingPointError("Nonfinite independent evaluation gain")
                mean = float(gain.mean())
                checked += len(gain)
        except (AssertionError, ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
            rows.append(dict(status="unavailable_reference", error=f"{type(exc).__name__}: {exc}"))
            continue
        values.append(mean)
        rows.append(dict(status="complete", mean_gain=mean))
    return dict(
        status="complete" if len(values) == len(events) else "incomplete",
        mean_gain=float(np.mean(values)) if values else None,
        rows=rows,
        density_checks=checked,
    )


def events(summary, external, point, control):
    qualified = summary["qualified"] and external["status"] == "complete"
    positive = summary["direction"] == "positive" if qualified else None
    agree = (positive == (external["mean_gain"] > 0)) if qualified else None
    correct = point["correct"] and (not positive if control else positive) if qualified else None
    return dict(qualified=summary["qualified"], positive=positive, agreement=agree, joint=correct)
