"""Prespecified truth-free initialization batches over public sklearn estimators."""

from dataclasses import replace

BATCHES = (("kmeans", 34), ("k-means++", 33), ("random_from_data", 33))


def batch_seed(seed, index):
    import numpy as np

    return (
        int(seed)
        if index == 0
        else int(np.random.SeedSequence([int(seed), index, 20261115]).generate_state(1)[0])
    )


def fit_diverse_batches(template, matrix):
    """Choose highest batch lower bound before native component-size checks.

    Every batch must converge. Numerical failure propagates; failed starts are
    never replaced. The first batch retains the native K seed and kmeans method.
    """
    import numpy as np
    from sklearn.base import clone

    models, records = [], []
    for i, (method, starts) in enumerate(BATCHES):
        seed = batch_seed(template.random_state, i)
        fitted = clone(template).set_params(n_init=starts, init_params=method, random_state=seed)
        fitted.fit(matrix)
        bound = float(fitted.lower_bound_)
        if not fitted.converged_ or not np.isfinite(bound):
            raise ValueError(f"Initialization batch {i} did not converge finitely")
        likelihood = float(fitted.score(matrix) * len(matrix))
        if not np.isfinite(likelihood):
            raise ValueError(f"Initialization batch {i} has nonfinite likelihood")
        models.append(fitted)
        records.append(
            dict(
                batch=i,
                method=method,
                starts=starts,
                seed=seed,
                lower_bound=bound,
                log_likelihood=likelihood,
                iterations=int(fitted.n_iter_),
                converged=True,
            )
        )
    winner = max(range(len(models)), key=lambda i: (records[i]["lower_bound"], -i))
    return models[winner], dict(
        policy="diverse_v1",
        total_starts=100,
        selected_batch=winner,
        batches=records,
        selection="Largest converged lower bound before size checks",
    )


def truth_free_model(model):
    from super_deli.models.integrative_config import IntegrativeConfig

    if type(model) is not IntegrativeConfig or model.fusion_policy != "retain_all_v1":
        raise ValueError("Use an explicit full-retention IntegrativeConfig")
    return replace(model, initialization_policy="diverse_v1", n_init=100)


def truth_free_analysis(config, *, arm_name):
    from super_deli.models.multimodal_analysis_config import MultimodalAnalysisConfig

    if type(config) is not MultimodalAnalysisConfig:
        raise ValueError("Use a MultimodalAnalysisConfig")
    chosen = [a for a in config.arms if a.name == arm_name]
    if len(chosen) != 1 or chosen[0].family != "integrative":
        raise ValueError("Select one named integrative arm")
    return replace(
        config,
        arms=tuple(
            replace(a, model=truth_free_model(a.model)) if a.name == arm_name else a
            for a in config.arms
        ),
    )
