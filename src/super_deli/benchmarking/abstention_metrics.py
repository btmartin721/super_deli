"""Coverage, withheld reasons and accepted errors with explicit empty denominators."""

import numpy as np

from super_deli.models.reference_abstention_config import DECISIONS


def metrics(y, result, *, known):
    y, predicted, accepted = np.asarray(y), result["predicted_indices"], result["accepted"]
    n = len(y)
    result_metrics = dict(
        n=n,
        accepted=int(accepted.sum()),
        accepted_fraction=float(accepted.mean()),
        reasons={s: int(np.count_nonzero(result["decisions"] == s)) for s in DECISIONS},
    )
    if not known:
        return result_metrics
    correct = predicted == y
    result_metrics.update(
        accepted_accuracy=float(correct[accepted].mean()) if accepted.any() else None,
        correct_accepted_fraction=float(np.mean(correct & accepted)),
        incorrect_accepted_fraction=float(np.mean(~correct & accepted)),
        known_withheld_fraction=float(np.mean(~accepted)),
        raw_accuracy=float(correct.mean()),
        per_class=[],
    )
    compatible = result.get("compatible")
    for c in range(result["probabilities"].shape[1]):
        rows = y == c
        chosen = rows & accepted
        result_metrics["per_class"].append(
            dict(
                reference=c,
                n=int(rows.sum()),
                accepted=int(chosen.sum()),
                accepted_accuracy=float(correct[chosen].mean()) if chosen.any() else None,
                true_reference_included=float(compatible[rows, c].mean())
                if compatible is not None and rows.any()
                else None,
            )
        )
    accuracies = [v["accepted_accuracy"] for v in result_metrics["per_class"]]
    result_metrics["accepted_balanced_accuracy"] = (
        float(np.mean(accuracies)) if all(v is not None for v in accuracies) else None
    )
    if compatible is not None:
        result_metrics["true_reference_included"] = float(compatible[np.arange(n), y].mean())
        result_metrics["mean_compatible_references"] = float(compatible.sum(1).mean())
    return result_metrics


def target(cohorts):
    known = cohorts["iid"]
    accuracy = known["accepted_accuracy"]
    return bool(
        known["accepted_fraction"] >= 0.8
        and accuracy is not None
        and accuracy >= 0.9
        and cohorts["near"]["accepted_fraction"] <= 0.1
        and cohorts["far"]["accepted_fraction"] <= 0.1
    )


def unscreened(p):
    chosen = p.argmax(1)
    return dict(
        probabilities=p,
        predicted_indices=chosen,
        accepted_indices=chosen,
        accepted=np.ones(len(p), dtype=bool),
        decisions=np.full(len(p), "assigned", dtype="U48"),
    )
