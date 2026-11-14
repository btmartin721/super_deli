"""Thin native workflow dispatch and ID-bound comparison extraction."""

from super_deli.core.provenance import configuration_hash, fingerprint_file
from super_deli.io.config import read_json


def run_matched_arm(arm, dataset, base_dir, directory):
    if arm.family == "pca_gmm":
        from super_deli.workflows.integrative import run_integrative

        run = run_integrative
    else:
        from super_deli.workflows.separate_gmm import run_separate_gmm

        run = run_separate_gmm
    return run(dataset, model_config=arm.model, base_dir=base_dir, output_dir=directory)


def read_matched_arm(arm, directory, cohort, frozen):
    import numpy as np

    summary = read_json(directory / "summary.json")
    native_config = (
        summary["provenance"]["configuration"]
        if arm.family == "pca_gmm"
        else summary["configuration"]
    )
    if (
        summary["status"] != "complete"
        or configuration_hash(native_config["model"]) != configuration_hash(arm.model.to_dict())
        or configuration_hash(native_config["dataset"]) != configuration_hash(frozen.to_dict())
    ):
        raise ValueError("Native model/dataset configuration mismatch")
    ids = summary["unit_ids"]
    wanted = cohort["unit_ids"]
    if len(ids) != len(set(ids)) or set(ids) != set(wanted):
        raise ValueError("Native fit does not retain the entire frozen cohort")
    order = [ids.index(u) for u in wanted]
    if arm.family == "pca_gmm":
        for view, expected in zip(summary["views"], cohort["views"], strict=True):
            members = {r["target_id"]: r["observation_ids"] for r in view["members"]}
            if (
                view["modality"] != expected["modality"]
                or view["feature_names"] != expected["feature_names"]
                or members
                != {u: [o] for u, o in zip(wanted, expected["observation_ids"], strict=True)}
            ):
                raise ValueError("Native PCA observation/feature alignment mismatch")
        path = directory / "mixture.npz"
        stability = summary["bootstrap"]
        preprocessing = summary["views"]
    else:
        native = read_json(directory / "cohort.json")
        if native != {k: cohort[k] for k in ("unit_ids", "views")}:
            raise ValueError("Native neural observation/feature alignment mismatch")
        path = directory / "point.npz"
        stability = summary["stability"]
        preprocessing = dict(
            configuration=arm.model.to_dict(), training=read_json(directory / "training.json")
        )
    expected_hash = (
        summary["provenance"]["mixture_sha256"]
        if arm.family == "pca_gmm"
        else summary["model_sha256"]
    )
    if fingerprint_file(path).sha256 != expected_hash:
        raise ValueError("Native mixture fingerprint mismatch")
    with np.load(path, allow_pickle=False) as a:
        labels = a["labels"][order].copy()
        scores = a["scores"][order].copy()
        membership = a["membership"][order].copy()
    k = summary["selected_k"]
    n = len(wanted)
    if (
        type(k) is not int
        or not 1 <= k <= arm.model.max_k
        or membership.shape != (n, k)
        or scores.ndim != 2
        or scores.shape[0] != n
        or scores.shape[1] != (arm.model.n_pcs if arm.family == "pca_gmm" else 4)
        or labels.shape != (n,)
        or labels.dtype.kind not in "iu"
        or not np.isfinite(scores).all()
        or not np.isfinite(membership).all()
        or np.any(membership < 0)
        or np.any(membership > 1)
        or not np.allclose(membership.sum(1), 1, rtol=1e-10, atol=1e-12)
        or not np.array_equal(labels, membership.argmax(1))
    ):
        raise ValueError("Malformed native mixture comparison arrays")
    return dict(
        summary=summary,
        point=dict(labels=labels.tolist(), scores=scores.tolist(), resolved=[True] * n),
        membership=membership.max(1).tolist(),
        stability=stability,
        preprocessing=preprocessing,
        candidates=read_json(directory / "candidates.json"),
    )
