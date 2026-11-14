"""Invented algebraic archive; zero optimizer/estimator or random-draw calls."""

import hashlib
import json
from pathlib import Path

import numpy as np

from super_deli.benchmarking.genomic_diagnosis_config import GenomicDiagnosisConfig


def sha(path):
    return hashlib.file_digest(Path(path).open("rb"), "sha256").hexdigest()


def fixture(root):
    root = Path(root)
    root.mkdir()
    base = root / "verification-evidence"
    base.mkdir()

    def write(relative, value):
        path = base / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2) + "\n")

    def save(relative, **arrays):
        path = base / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **arrays)

    case = dict(case_id="invented-main", law="common_hwe", n=16, n_subsamples=1)
    prefix = "main/invented-main"
    analysis = prefix + "/analysis"
    raw = np.tile(
        np.array([[0, 0, 0, 0], [2, 0, 2, 0], [0, 2, 0, 2], [2, 2, 2, 2]], dtype=float), (4, 1)
    )
    samples = [f"invented_sample_{i:02d}" for i in range(16)]
    units = [f"invented_unit_{i:02d}" for i in range(16)]
    source = base / analysis / "inputs/source/invented.phy"
    source.parent.mkdir(parents=True)
    source.write_text(
        "16 4\n"
        + "".join(
            s + " " + "".join("A" if g == 0 else "G" for g in row) + "\n"
            for s, row in zip(samples, raw, strict=True)
        )
    )
    keys = [f"{sha(source)}:{j}" for j in range(4)]
    for name, array in (("dosages", raw), ("source-alleles", np.where(raw == 0, "A", "G"))):
        np.save(base / analysis / "inputs" / (name + ".npy"), array, allow_pickle=False)
    write(
        analysis + "/inputs/sources.json",
        [dict(snapshot="source/invented.phy", source=dict(sha256=sha(source)))],
    )
    write(
        analysis + "/inputs/cohort.json",
        dict(
            unit_ids=units,
            sample_ids=samples,
            locus_keys=keys,
            original_locus_indices=list(range(4)),
        ),
    )
    save(prefix + "/generating-law.npz", values=raw, truth=np.zeros(16, dtype=int))
    write(prefix + "/case.json", case)
    arms = dict(
        pca_diagonal="pca_diagonal",
        pca_shared="pca_shared",
        neural_seed_0="categorical_ae",
        neural_seed_1="categorical_ae",
        neural_seed_2="categorical_ae",
    )

    def model(relative, values, family):
        from super_deli.benchmarking.genomic_diagnosis_math import forward

        mean = values.mean(0)
        scale = np.sqrt(2 * (mean / 2) * (1 - mean / 2))
        x = (values - mean) / scale
        a = dict(
            kept_indices=np.arange(4),
            means=mean,
            scales=scale,
            source_locus_count=np.asarray(4),
            diagonal=np.asarray(family == "pca_diagonal"),
        )
        if family == "categorical_ae":
            for side, widths in (("encoder", (4, 16, 8, 2)), ("decoder", (2, 8, 16, 12))):
                for i in range(3):
                    a[f"{side}_{i}_weight"] = np.zeros((widths[i + 1], widths[i]))
                    a[f"{side}_{i}_bias"] = np.zeros(widths[i + 1])
            for i in range(3):
                a[f"encoder_{i}_weight"][0, 0] = 0.3
                a[f"encoder_{i}_weight"][1, 1] = 0.3
            latent = forward(x, a, "encoder")[0]
            a["losses"] = np.repeat(np.log(3), 501)
        else:
            a.update(
                pca_mean=x.mean(0),
                pca_components=np.array([[1, 0, 1, 0], [0, 1, 0, 1]]) / np.sqrt(2),
            )
            latent = (x - a["pca_mean"]) @ a["pca_components"].T
        a.update(
            latent_means=np.zeros(2) if family == "pca_diagonal" else latent.mean(0),
            latent_scales=np.ones(2) if family == "pca_diagonal" else latent.std(0),
        )
        scores = (latent - a["latent_means"]) / a["latent_scales"]
        covariance = (
            scores.var(0)[None] + 1e-6
            if family == "pca_diagonal"
            else scores.T @ scores / len(scores) + np.eye(2) * 1e-6
        )
        a.update(
            scores=scores,
            labels=np.zeros(len(values), dtype=int),
            membership=np.ones((len(values), 1)),
            mixture_weights=np.ones(1),
            mixture_means=np.zeros((1, 2)),
            mixture_covariance=covariance,
        )
        if family == "pca_diagonal":
            ll = (
                -0.5
                * (
                    2 * np.log(2 * np.pi)
                    + np.log(covariance).sum()
                    + (scores**2 / covariance).sum(1)
                ).sum()
            )
        else:
            ll = (
                -0.5
                * (
                    2 * np.log(2 * np.pi)
                    + np.linalg.slogdet(covariance)[1]
                    + np.einsum("ni,ij,nj->n", scores, np.linalg.inv(covariance), scores)
                ).sum()
            )
        parameters = 4 if family == "pca_diagonal" else 5
        candidates = [
            dict(
                k=1,
                status="eligible",
                bic=float(-2 * ll + parameters * np.log(len(values))),
                converged=True,
                hard_cluster_sizes=[len(values)],
                iterations=1,
            )
        ]
        save(relative + "/model.npz", **a)
        write(
            relative + "/model.json",
            dict(
                schema_version=1,
                family=family,
                locus_keys=keys,
                model_sha256=sha(base / relative / "model.npz"),
                candidates=candidates,
                diagnostics={},
            ),
        )

    bundle = {}
    indices = np.arange(12)
    pair = np.zeros((16, 16), dtype=int)
    pair[np.ix_(indices, indices)] = 1
    for arm, family in arms.items():
        model(analysis + "/arms/" + arm, raw, family)
        model(analysis + "/subsamples/draw-0000/" + arm, raw[indices], family)
        record = dict(
            draw=0,
            training_indices=indices.tolist(),
            unit_ids=units[:12],
            status="complete",
            labels=[0] * 12,
            selected_k=1,
            error=None,
        )
        bundle[arm] = record
        write(analysis + "/arms/" + arm + "/subsamples.json", [record])
        save(
            analysis + "/arms/" + arm + "/stability.npz",
            pair_included=pair,
            pair_successful=pair,
            pair_coassigned=pair,
        )
    write(analysis + "/subsamples/draw-0000/records.json", bundle)
    write(analysis + "/manifest.json", dict(model=dict(min_cluster_size=3)))
    write(
        analysis + "/summary.json",
        dict(
            status="complete",
            inference=dict(status="not_computed", p_value=None),
            unit_ids=units,
            models=[dict(name=n, family=f) for n, f in arms.items()],
        ),
    )
    write(
        prefix + "/completion.json",
        dict(status="complete", audit=dict(passed=True), steps=dict(passed=True)),
    )
    write("study-freeze.json", dict(cases=dict(preflight=[], main=[case])))
    (root / "completion.json").write_text(
        json.dumps(dict(completed=True, head_sha="0" * 40, version="invented-fixture"))
    )
    inventory = {
        p.relative_to(base).as_posix(): dict(bytes=p.stat().st_size, sha256=sha(p))
        for p in sorted(base.rglob("*"))
        if p.is_file()
    }
    (root / "verification-manifest.json").write_text(json.dumps(inventory, indent=2))
    return GenomicDiagnosisConfig(
        sha(root / "completion.json"),
        sha(root / "verification-manifest.json"),
        "0" * 40,
        source_role="invented_fixture",
        source_version="invented-fixture",
        expected_observations=1,
        expected_point_models=5,
        expected_subsample_models=5,
    )
