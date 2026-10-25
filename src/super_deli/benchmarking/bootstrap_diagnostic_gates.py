"""Frozen-parameter gate accounting; counterfactuals are never production selections."""

import math


def unique_support(indices):
    return list(dict.fromkeys(map(int, indices)))


def truth_sizes(labels, indices):
    if labels is None:
        return None
    return [len({i for i in indices if labels[i] == k}) for k in range(4)]


def validation_gain(k, folds, *, unique=False):
    terms = []
    for f in folds:
        logs = f.get("log_densities", {})
        if str(k) not in logs or "1" not in logs:
            return None
        ids = f["original_validation_indices"]
        a, b = logs[str(k)], logs["1"]
        if len(a) != len(ids) or len(b) != len(ids):
            raise ValueError("Validation densities must retain every occurrence")
        take = list(range(len(ids)))
        if unique:
            take = list({u: j for j, u in reversed(list(enumerate(ids)))}.values())
        terms.extend(a[j] - b[j] for j in take)
    return math.fsum(terms) / len(terms) if terms and all(map(math.isfinite, terms)) else None


def gates(full, folds, indices, selection, *, minimum, truth=None):
    if selection.get("status") != "complete":
        return dict(
            status="unavailable",
            selected_k=None,
            error=selection.get("error"),
            candidates=[],
            counterfactuals={},
        )
    if len(folds) != 5 or any(f["status"] != "complete" for f in folds):
        raise ValueError("Available screen cannot contain an unresolved fold")
    rows = []
    for c in full:
        k = c["k"]
        others = [next(r for r in f["candidates"] if r["k"] == k) for f in folds]
        n = [len(set(indices))] + [len(set(f["original_training_indices"])) for f in folds]
        fitted = all(r["status"] != "insufficient_samples" for r in [c, *others])
        full_ok = c["status"] == "eligible"
        fold_ok = all(r["status"] == "eligible" for r in others)
        gain = validation_gain(k, folds) if fitted else None
        equal = validation_gain(k, folds, unique=True) if fitted else None
        positive = k == 1 or gain is not None and gain > 0
        feasible = all(v >= k * minimum for v in n)
        first = (
            "distinct_feasibility"
            if not feasible
            else "full_eligibility"
            if not full_ok
            else "fold_eligibility"
            if not fold_ok
            else "nonpositive_gain"
            if not positive
            else "selected"
            if k == selection["screened_k"]
            else "aic_ranking"
        )
        rows.append(
            dict(
                k=k,
                full_status=c["status"],
                fold_statuses=[r["status"] for r in others],
                full_distinct_feasible=n[0] >= k * minimum,
                all_fold_distinct_feasible=all(v >= k * minimum for v in n[1:]),
                full_eligible=full_ok,
                all_folds_eligible=fold_ok,
                positive_gain=positive,
                pooled_validation_gain=gain,
                equal_original_validation_gain=equal,
                fitted_in_all_searches=fitted,
                passes=full_ok and fold_ok and positive,
                first_failing_gate=first,
                full_aic=c.get("aic"),
                full_hard_sizes=c.get("hard_cluster_sizes"),
                full_distinct_sizes=c.get("distinct_unit_sizes"),
                fold_hard_sizes=[r.get("hard_cluster_sizes") for r in others],
                fold_distinct_sizes=[r.get("distinct_unit_sizes") for r in others],
                generating_full_sizes=truth_sizes(truth, indices),
                generating_fold_sizes=[
                    truth_sizes(truth, f["original_training_indices"]) for f in folds
                ],
                full_and_fold_distinct_counts=n,
            )
        )
    allowed = [r for r in rows if r["passes"]]
    chosen = min(allowed, key=lambda r: (r["full_aic"], r["k"]))["k"]
    if chosen != selection["screened_k"]:
        raise ValueError("Independent gate reconstruction differs from stored selection")
    for actual in selection["candidates"]:
        row = next(r for r in rows if r["k"] == actual["k"])
        if actual["passes"] != (actual["k"] == 1 or row["passes"]):
            raise ValueError("Candidate gate differs from original screen")
        if actual["pooled_validation_gain"] is not None:
            if not math.isclose(
                actual["pooled_validation_gain"],
                row["pooled_validation_gain"],
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise ValueError("Pooled validation gain differs")
    counter = {}
    for policy in (
        "remove_full_size",
        "remove_fold_size",
        "remove_gain",
        "occurrence_sizes",
        "equal_validation_weights",
    ):
        missing = []
        accepted = []
        for row, c in zip(rows, full, strict=True):
            if (
                policy in {"remove_full_size", "remove_fold_size", "occurrence_sizes"}
                and not row["fitted_in_all_searches"]
            ):
                missing.append(row["k"])
                continue
            full_ok = row["full_eligible"] or policy == "remove_full_size"
            fold_ok = row["all_folds_eligible"] or policy == "remove_fold_size"
            gain = (
                row["equal_original_validation_gain"]
                if policy == "equal_validation_weights"
                else row["pooled_validation_gain"]
            )
            if policy == "occurrence_sizes":
                full_ok = min(row["full_hard_sizes"]) >= minimum
                fold_ok = all(min(s) >= minimum for s in row["fold_hard_sizes"])
            positive = row["k"] == 1 or policy == "remove_gain" or gain is not None and gain > 0
            if full_ok and fold_ok and positive:
                accepted.append(c)
        counter[policy] = (
            dict(status="unavailable", missing_candidates=missing, selected_k=None)
            if missing
            else dict(
                status="available", selected_k=min(accepted, key=lambda c: (c["aic"], c["k"]))["k"]
            )
        )
    return dict(
        status="available",
        selected_k=chosen,
        candidates=rows,
        counterfactuals=counter,
        scope="Frozen-model diagnostic counterfactuals; never selection or species support",
    )
