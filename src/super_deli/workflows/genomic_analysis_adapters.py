"""Native dispatch and strict ID-bound extraction; no estimator changes."""

from super_deli.core.provenance import configuration_hash, fingerprint_file
from super_deli.io.config import read_json


def run_genomic_arm(arm, dataset, base_dir, directory):
    if arm.family == "shape":
        from super_deli.workflows.shape_gmm import run_shape_gmm

        return run_shape_gmm(
            dataset,
            model_config=arm.model,
            base_dir=base_dir,
            output_dir=directory,
            modality_name=arm.modalities[0],
        )
    if arm.family == "genomic":
        from super_deli.workflows.genomic_baseline import run_genomic_baseline

        return run_genomic_baseline(
            dataset,
            model_config=arm.model,
            base_dir=base_dir,
            output_dir=directory,
            modality_name=arm.modalities[0],
            block_ids=arm.block_ids,
        )
    if arm.family == "tabular":
        from super_deli.workflows.tabular_gmm import run_tabular_gmm

        return run_tabular_gmm(
            dataset,
            model_config=arm.model,
            base_dir=base_dir,
            output_dir=directory,
            modality_name=arm.modalities[0],
        )
    from super_deli.workflows.integrative import run_integrative

    return run_integrative(dataset, model_config=arm.model, base_dir=base_dir, output_dir=directory)


def read_genomic_arm(arm, directory, cohort, frozen):
    import numpy as np

    s = read_json(directory / "summary.json")
    c = s["provenance"]["configuration"]
    if (
        s["status"] != "complete"
        or configuration_hash(c["model"]) != configuration_hash(arm.model.to_dict())
        or configuration_hash(c["dataset"]) != configuration_hash(frozen.to_dict())
    ):
        raise ValueError("Native configuration mismatch")
    ids = s["specimen_ids"] if arm.family == "genomic" else s["unit_ids"]
    wanted = cohort["unit_ids"]
    if len(ids) != len(set(ids)) or set(ids) != set(wanted):
        raise ValueError("Native arm did not retain the complete frozen cohort")
    lookup = {u: i for i, u in enumerate(ids)}
    order = [lookup[u] for u in wanted]
    if arm.family == "genomic":
        expected_blocks = (
            None if arm.block_ids is None else dict(schema_version=1, block_ids=list(arm.block_ids))
        )
        if c["blocks"] != expected_blocks or c["modality"] != arm.modalities[0]:
            raise ValueError("Native genomic block/modality declaration mismatch")
        for key, file in (
            ("labels", "labels.npy"),
            ("scores", "pca_scores.npy"),
            ("membership", "membership.npy"),
        ):
            if key == "labels":
                arrays = {}
            arrays[key] = np.load(directory / file, allow_pickle=False)[order]
        stability = s["stability"]
        failure_count = len(stability["failures"])
        evidence_type = "source_locus_block_bootstrap_conditional_on_specimens"
        candidates = s["candidates"]
        preprocessing = s["preprocessing"]
        covariance = "diag"
    else:
        expected = s["provenance"][
            "mixture_sha256" if arm.family == "integrative" else "mixture_array_sha256"
        ]
        if fingerprint_file(directory / "mixture.npz").sha256 != expected:
            raise ValueError("Native mixture fingerprint mismatch")
        with np.load(directory / "mixture.npz", allow_pickle=False) as archive:
            arrays = {key: archive[key][order].copy() for key in ("labels", "scores", "membership")}
        stability = s["bootstrap"]
        failure_count = stability["failed"]
        evidence_type = (
            "paired_specimen_bootstrap_fixed_source_loci"
            if arm.family == "integrative"
            else "specimen_row_bootstrap"
        )
        candidates = read_json(directory / "candidates.json")
        preprocessing = s["views"] if arm.family == "integrative" else s.get("preprocessing")
        covariance = s["covariance_type"]
        if arm.family == "shape":
            from super_deli.io.shape_model import read_shape_model

            read_shape_model(directory / "shape")
            shape_summary = read_json(directory / "shape" / "summary.json")
            preprocessing = {
                key: value for key, value in shape_summary.items() if key != "provenance"
            }
            evidence_type = "whole_specimen_bootstrap_refitting_gpa_and_shape_pca"
    labels, scores, membership = (arrays[k] for k in ("labels", "scores", "membership"))
    n, k = len(wanted), s["selected_k"]
    if (
        type(k) is not int
        or not 1 <= k <= arm.model.max_k
        or labels.shape != (n,)
        or labels.dtype.kind not in "iu"
        or scores.shape != (n, arm.model.n_pcs)
        or not np.isfinite(scores).all()
        or membership.shape != (n, k)
        or not np.isfinite(membership).all()
        or np.any(membership < 0)
        or np.any(membership > 1)
        or not np.allclose(membership.sum(1), 1, rtol=1e-10, atol=1e-12)
        or not np.array_equal(labels, membership.argmax(1))
    ):
        raise ValueError("Malformed native comparison arrays")
    result = dict(
        point=dict(labels=labels.tolist(), scores=scores.tolist(), resolved=[True] * n),
        membership=membership.max(1).tolist(),
        selected_k=k,
        status="partial" if failure_count else "complete",
        stability=stability,
        evidence_type=evidence_type,
        candidates=candidates,
        preprocessing=preprocessing,
        covariance_type=covariance,
    )
    if arm.family == "integrative" and arm.model.fusion_policy != "joint_pca":
        if s.get("fusion_policy") != arm.model.fusion_policy:
            raise ValueError("Native fusion policy differs from the declared arm")
        result.update(fusion_policy=s["fusion_policy"], retained_dimensions=arm.model.n_pcs)
    return result
