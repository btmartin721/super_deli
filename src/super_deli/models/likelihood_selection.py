"""Prespecified component selection and global departure statistics on one fit pool."""


def candidate_statistics(candidates):
    from super_deli.models.gaussian_calibration import bic_improvement

    eligible = [c for c in candidates if c["status"] == "eligible"]
    one = next(c for c in eligible if c["k"] == 1)
    return {
        "statistic": 2 * (max(c["log_likelihood"] for c in eligible) - one["log_likelihood"]),
        "selected_k": min(eligible, key=lambda c: (c["aic"], c["k"]))["k"],
        "bic_statistic": bic_improvement(candidates),
        "bic_selected_k": min(eligible, key=lambda c: (c["bic"], c["k"]))["k"],
    }
