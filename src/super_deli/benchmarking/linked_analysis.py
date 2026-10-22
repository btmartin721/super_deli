"""Independent categorical likelihood, donor provenance and genomic pipeline audit."""

import io
import json
from collections import Counter
from pathlib import Path

from super_deli.benchmarking.empirical_analysis import audit_fitted_study
from super_deli.benchmarking.genomic_analysis import _equivalent
from super_deli.benchmarking.genomic_analysis import point as genomic_point
from super_deli.benchmarking.linked_study import case_policy, simulate_linked, summarize_linked
from super_deli.io.config import read_json
from super_deli.models.linked_config import CASES, LinkedStudyConfig
from super_deli.workflows.linked_calibration import NAMESPACE, TAIL_OPTIONS, TARGET


def verify_model(base, output, values, layout, policy):
    import numpy as np

    fitted = read_json(output / "layout.json")
    for key in ("sample_ids", "unit_ids", "locus_indices", "modality"):
        if fitted[key] != layout[key]:
            raise ValueError("Linked source identity layout mismatch")
    if fitted["block_ids"] != list(policy.block_ids):
        raise ValueError("Fitted block map differs from declaration")
    mapping = {}
    for j, name in enumerate(policy.block_ids):
        mapping.setdefault(name, []).append(j)
    if len(fitted["blocks"]) != len(mapping):
        raise ValueError("Fitted block count mismatch")
    nll = 0.0
    with np.load(output / "null_model.npz", allow_pickle=False) as arrays:
        x = arrays["values"].copy()
        _equivalent(values, x)
        for b, ((name, indices), block) in enumerate(
            zip(mapping.items(), fitted["blocks"], strict=True)
        ):
            observed = [tuple(map(int, row)) for row in x[:, indices]]
            counts = Counter(observed)
            patterns = sorted(counts)
            expected_counts = np.array([counts[p] for p in patterns])
            probabilities = expected_counts / len(x)
            prefix = f"block_{b:04d}"
            np.testing.assert_array_equal(arrays[prefix + "_profiles"], patterns)
            np.testing.assert_array_equal(arrays[prefix + "_counts"], expected_counts)
            np.testing.assert_array_equal(arrays[prefix + "_probabilities"], probabilities)
            lookup = {p: i for i, p in enumerate(patterns)}
            np.testing.assert_array_equal(
                arrays[prefix + "_profile_ids"], [lookup[p] for p in observed]
            )
            expected = {
                "block_id": name,
                "locus_indices": indices,
                "n_profiles": len(patterns),
                "singleton_profiles": sum(c == 1 for c in counts.values()),
                "singleton_specimen_fraction": sum(c == 1 for c in counts.values()) / len(x),
                "maximum_profile_probability": max(counts.values()) / len(x),
            }
            if block != expected:
                raise ValueError("Empirical block support diagnostics disagree")
            # Ordered specimen likelihood, independently evaluated one row at a time.
            nll -= sum(np.log(counts[p] / len(x)) for p in observed)
    if not np.isclose(nll, fitted["fit"]["negative_log_likelihood"], atol=1e-9, rtol=1e-12):
        raise ValueError("Independent categorical likelihood mismatch")
    if fitted["fit"]["n_blocks"] != len(mapping) or fitted["fit"]["n_observed_profiles"] != sum(
        b["n_profiles"] for b in fitted["blocks"]
    ):
        raise ValueError("Fitted profile totals disagree")

    def draw(seed):
        donors = np.random.default_rng(seed).integers(len(x), size=(len(x), len(mapping)))
        generated = np.empty_like(x)
        for b, indices in enumerate(mapping.values()):
            generated[:, indices] = x[donors[:, b]][:, indices]
        return {"genotypes": generated, "donor_indices": donors}

    return fitted, draw, *x.shape


def point(archive, record, policy, values, layout):
    import numpy as np

    draw = isinstance(values, dict)
    genomic_point(archive, record, policy, values["genotypes"] if draw else values, layout)
    prefix = record["archive_prefix"]
    if draw:
        with io.BytesIO(archive.read(prefix + "/raw/donor_indices.npy")) as handle:
            np.testing.assert_array_equal(
                np.load(handle, allow_pickle=False), values["donor_indices"]
            )
    if record["status"] == "complete":
        run = json.loads(archive.read(prefix + "/run/summary.json"))
        if run["provenance"]["configuration"]["blocks"] != {
            "schema_version": 1,
            "block_ids": list(policy.block_ids),
        }:
            raise ValueError("Point pipeline did not use the declared block map")


def audit_linked_study(root, *, output_dir):
    from super_deli.models.linked_null import write_raw_linked
    from super_deli.workflows.linked_calibration import _dependencies

    _dependencies(
        LinkedStudyConfig.from_dict(
            read_json(Path(root) / CASES[0] / "study.json")
        ).calibration.block_ids
    )
    backend = {
        "cases": CASES,
        "study_type": LinkedStudyConfig,
        "policy_for": case_policy,
        "summarize": summarize_linked,
        "simulator": simulate_linked,
        "point_auditor": point,
        "tail_options": TAIL_OPTIONS,
        "namespace": NAMESPACE,
        "decision_key": "reject_independent_blocks",
        "target": TARGET,
        "writer": write_raw_linked,
        "verify_model": verify_model,
        "audit_scope": (
            "Regenerated all raw observations/draws and donor indices; independently "
            "checked categorical profile MLEs/ordered-row likelihood, block "
            "maps/support, every archive hash, Patterson filtering/scaling, PCA "
            "eigensystem/projection, selected diagonal-GMM likelihood/BIC/membership,"
            " seeds/selection/tails/denominators. No ingestion or clustering refits. "
            "Nonselected likelihoods are retained/checksummed; outer failures retain "
            "counts without a complete numerical audit."
        ),
    }
    return audit_fitted_study(root, output_dir=output_dir, backend=backend)
