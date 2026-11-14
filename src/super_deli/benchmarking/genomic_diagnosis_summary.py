"""Parent-level descriptive aggregation and a prespecified development decision."""

from collections import Counter

import numpy as np


def adjusted_rand(left, right):
    _, a = np.unique(left, return_inverse=True)
    _, b = np.unique(right, return_inverse=True)
    table = np.zeros((a.max() + 1, b.max() + 1), int)
    np.add.at(table, (a, b), 1)

    def choose(x):
        return np.sum(x * (x - 1) / 2)

    pairs = len(a) * (len(a) - 1) / 2
    cell = choose(table)
    rows = choose(table.sum(1))
    cols = choose(table.sum(0))
    expected = rows * cols / pairs
    denominator = 0.5 * (rows + cols) - expected
    return float((cell - expected) / denominator) if denominator else 1.0


def summarize(records, role):
    primary = [r for r in records if r["phase"] == "main" and r["draw"] is None]
    cells = []
    for key in sorted({(r["law"], r["n"], r["arm"]) for r in primary}):
        rows = [r for r in primary if (r["law"], r["n"], r["arm"]) == key]

        def mean(name, rows=rows):
            values = [r[name] for r in rows if r[name] is not None]
            return float(np.mean(values)) if values else None

        cells.append(
            dict(
                law=key[0],
                n=key[1],
                arm=key[2],
                independent_datasets=len(rows),
                k_counts=dict(Counter(str(r["selected_k"]) for r in rows)),
                selected_bic_improvement_mean=mean("selected_bic_improvement"),
                twice_likelihood_gain_mean=mean("twice_likelihood_gain"),
                parameter_cost_mean=mean("parameter_cost"),
                ari_mean=mean("ari"),
                rare_sensitivity_share_mean=mean("rare_sensitivity_share"),
                first_hidden_saturation_mean=float(
                    np.mean([r["hidden_saturation"][0] for r in rows])
                )
                if rows[0]["hidden_saturation"]
                else None,
                associations={
                    name: dict(
                        radius_spearman_mean=float(
                            np.mean(
                                [
                                    r["burden_associations"][name]["radius_spearman"]
                                    for r in rows
                                    if r["burden_associations"][name]["radius_spearman"] is not None
                                ]
                            )
                        )
                        if any(
                            r["burden_associations"][name]["radius_spearman"] is not None
                            for r in rows
                        )
                        else None,
                        label_variance_fraction_mean=float(
                            np.mean(
                                [
                                    r["burden_associations"][name]["label_variance_fraction"]
                                    for r in rows
                                    if r["burden_associations"][name]["label_variance_fraction"]
                                    is not None
                                ]
                            )
                        )
                        if any(
                            r["burden_associations"][name]["label_variance_fraction"] is not None
                            for r in rows
                        )
                        else None,
                    )
                    for name in rows[0]["burden_associations"]
                },
            )
        )
    paired = []
    for arm in sorted({r["arm"] for r in primary if r["family"] == "categorical_ae"}):
        control = []
        alternative = []
        for row in [r for r in primary if r["arm"] == arm]:
            reference = next(
                r for r in primary if r["case_id"] == row["case_id"] and r["arm"] == "pca_shared"
            )
            if row["law"] in ("common_hwe", "rare_complete_hwe"):
                control.append((row["selected_k"] > 1, reference["selected_k"] > 1))
            elif row["law"] == "four_groups":
                alternative.append((row["ari"], reference["ari"]))
        paired.append(
            dict(
                arm=arm,
                control_datasets=len(control),
                neural_control_splits=sum(a for a, b in control),
                pca_control_splits=sum(b for a, b in control),
                neural_only_control_splits=sum(a and not b for a, b in control),
                pca_only_control_splits=sum(b and not a for a, b in control),
                alternative_datasets=len(alternative),
                neural_ari_better=sum(a > b + 1e-12 for a, b in alternative),
                pca_ari_better=sum(b > a + 1e-12 for a, b in alternative),
            )
        )
    if role == "invented_fixture":
        decision = "invented_fixture_no_scientific_decision"
    elif all(
        r["neural_ari_better"] == 0
        and r["neural_only_control_splits"] > r["pca_only_control_splits"]
        for r in paired
    ):
        decision = "no_go_for_promotion_unguided_tuning_or_large_calibration"
    else:
        decision = "insufficient_basis_for_promotion_review_before_new_work"
    return dict(
        cells=cells,
        paired_main_comparisons=paired,
        decision=dict(
            status=decision,
            retained_as="experimental",
            automatic_default_change=False,
            implementation_fault_detected=False,
            interpretation=(
                "Integrity checks passed. Associations do not establish a causal mechanism. "
                "No numerical repair is justified by this audit; future development requires "
                "a prespecified independent benefit/control validation."
            ),
        ),
        primary_independent_datasets=len({r["case_id"] for r in primary}),
        scope=(
            "Main datasets only for operating comparisons; preflight and subsamples remain "
            "secondary repeated measurements. Heterogeneous controls are paired descriptive "
            "totals, not one binomial false-positive estimate."
        ),
    )
