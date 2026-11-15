"""Bounded point contrasts; generating labels never enter fitting or arm selection."""

from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist

from super_deli.benchmarking.multimodal_validation import equal_point_arrays
from super_deli.benchmarking.signal_loss_config import PAIRS
from super_deli.benchmarking.signal_loss_fit import (
    audit_candidates,
    fit_scores,
    observe_candidates,
    score_matrix,
)
from super_deli.benchmarking.signal_loss_metrics import costs, geometry, projected_energy, recovery
from super_deli.benchmarking.signal_loss_trace import LocalReader, public_trace, read_context
from super_deli.io.config import read_json, write_json
from super_deli.workflows.genomic_analysis_adapters import run_genomic_arm


def point_seed():
    return int(np.random.SeedSequence(301).spawn(1)[0].generate_state(1)[0])


def run_case(
    dataset,
    analysis,
    *,
    base_dir,
    directory,
    truth_path,
    law,
    origin,
    source_reader=None,
    source_entry=None,
):
    directory = Path(directory)
    directory.mkdir()
    native_root = directory / "native"
    native_root.mkdir()
    selected = [a for a in analysis.arms if a.name in {"genomic", "shape", "combined"}]
    if {a.name for a in selected} != {"genomic", "shape", "combined"}:
        raise ValueError("Three native source arms required")
    native_rows = []
    audited = 0
    parity = 0
    for arm in selected:
        arm = replace(arm, model=replace(arm.model, n_bootstrap=0))
        try:
            with observe_candidates(directory / ("native-candidates-" + arm.name)):
                run_genomic_arm(arm, dataset, base_dir, native_root / arm.name)
            status = "complete"
            error = None
        except Exception as exc:
            status = "failed"
            error = f"{type(exc).__name__}: {exc}"
        audited += audit_candidates(directory / ("native-candidates-" + arm.name))
        native_rows.append(dict(name=arm.name, status=status, error=error))
        if status == "complete" and source_reader is not None:
            old = Path(source_entry["analysis"]) / "arms" / arm.name
            # Verify every old numeric point array before allowing a native parity claim.
            for key in source_reader.inventory:
                if key.startswith(old.as_posix() + "/") and key.endswith((".npz", ".npy")):
                    if not Path(key).name.startswith(("stability", "coassignment")):
                        source_reader.path(key)
            parity += equal_point_arrays(source_reader.root / old, native_root / arm.name)
    if any(r["status"] != "complete" for r in native_rows):
        result = dict(
            status="failed",
            origin=origin,
            law=law,
            variants=[],
            native=native_rows,
            error="A required parent point failed; cached contrasts withheld",
            audited_candidates=audited,
        )
        write_json(result, directory / "summary.json")
        return result
    reader = LocalReader(native_root)
    # Metadata and labels are evaluation-only; none are passed to fit_scores.
    reader_truth = directory / "truth-evaluation-only.json"
    write_json(read_json(truth_path), reader_truth)
    # read_context uses one reader for all paths, so place a label-only copy beside native fits.
    write_json(read_json(truth_path), native_root / "truth-evaluation-only.json")
    context = read_context(reader, "combined", "genomic", "shape", "truth-evaluation-only.json")
    y = context["labels"]
    parent = next(a for a in analysis.arms if a.name == "combined").model
    if parent.n_pcs != 2 or parent.min_cluster_size != 5 or parent.covariance_type != "tied":
        raise ValueError("Frozen native baseline settings required")
    seed = point_seed()
    rows = [
        dict(
            name="baseline",
            status="complete",
            selected_k=context["summary"]["selected_k"],
            dimension=2,
            minimum=5,
            covariance_type="tied",
            **recovery(y, context["point"]["labels"], law),
            costs=context["costs"],
            shape_energy_retained=context["joint_view_centroid_energy_retained"]["outline"],
        )
    ]
    variants = []
    names = tuple(context["views"])
    for d in (2, 4, 8):
        config = replace(parent, n_pcs=d, min_cluster_size=9, n_bootstrap=0)
        variants.append((f"all_pc{d}_min9", names, d, config, "tied"))
    for pair in PAIRS:
        config = replace(
            parent,
            views=tuple(next(v for v in parent.views if v.modality == name) for name in pair),
            n_pcs=2,
            min_cluster_size=5,
            n_bootstrap=0,
        )
        variants.append(("pair_" + "_".join(pair), pair, 2, config, "tied"))
    diagnostic = directory / "diagnostics"
    diagnostic.mkdir()
    for name, chosen, d, config, covariance in variants:
        destination = diagnostic / name
        try:
            fused, scores, components = score_matrix(context, chosen, d)
            if name == "all_pc2_min9":
                np.testing.assert_array_equal(scores, context["scores"])
            if d == 8:
                np.testing.assert_allclose(pdist(fused), pdist(scores), rtol=1e-9, atol=1e-10)
            result = fit_scores(scores, config, seed, covariance, destination)
            with (destination / "representation.npz").open("xb") as f:
                np.savez(
                    f,
                    fused=fused,
                    scores=scores,
                    components=components,
                    selected_views=np.asarray(chosen),
                )
            row = dict(name=name, **result)
            if result["status"] == "complete":
                with np.load(destination / "point.npz", allow_pickle=False) as a:
                    row.update(recovery(y, a["labels"], law))
                row.update(
                    geometry=geometry(scores, y),
                    costs=costs(result["candidates"], len(y), d, covariance),
                )
                if "outline" in chosen:
                    offset = 2 * chosen.index("outline")
                    isolated = np.zeros_like(fused)
                    isolated[:, offset : offset + 2] = fused[:, offset : offset + 2]
                    row["shape_energy_retained"] = projected_energy(isolated, y, components)
                    if d == 8:
                        np.testing.assert_allclose(row["shape_energy_retained"], 1, atol=1e-9)
            audited += result["audited_candidates"]
        except AssertionError:
            raise
        except Exception as exc:
            # Preserve an infeasible or failed contrast without substituting another arm.
            row = dict(
                name=name, status="failed", selected_k=None, error=f"{type(exc).__name__}: {exc}"
            )
            if not destination.exists():
                destination.mkdir()
            write_json(row, destination / "failure.json")
        rows.append(row)
    for kind, families in (("genomic", ("diag", "tied")), ("shape", ("full", "tied"))):
        source = context["single"][kind]
        arm = next(a for a in selected if a.name == kind)
        config = replace(arm.model, n_bootstrap=0)
        for family in families:
            name = kind + "_" + family + "_common_order"
            destination = diagnostic / name
            result = fit_scores(source["scores"], config, seed, family, destination)
            row = dict(name=name, **result)
            if result["status"] == "complete":
                with np.load(destination / "point.npz", allow_pickle=False) as a:
                    row.update(recovery(y, a["labels"], law))
                    row["same_as_source_partition"] = bool(
                        np.array_equal(
                            np.equal.outer(a["labels"], a["labels"]),
                            np.equal.outer(source["labels"], source["labels"]),
                        )
                    )
                row["costs"] = costs(result["candidates"], len(y), 2, family)
            audited += result["audited_candidates"]
            rows.append(row)
    # Persist truth only after fixed unsupervised fits; keep individual rows local.
    with (directory / "context.npz").open("xb") as f:
        np.savez(
            f,
            unit_ids=np.asarray(context["unit_ids"]),
            labels=y,
            fused=context["fused"],
            joint_scores=context["scores"],
            **{name + "_profiles": v["profiles"] for name, v in context["views"].items()},
        )
    write_json(public_trace(context), directory / "trace.json")
    result = dict(
        status="complete" if all(r["status"] == "complete" for r in rows) else "partial",
        origin=origin,
        law=law,
        n=len(y),
        native=native_rows,
        variants=rows,
        exact_parity_arrays=parity,
        audited_candidates=audited,
        planned_searches=15,
        native_workflows=3,
        score_searches=12,
        conditioning=(
            "fixed native training transforms and parent unit order; no resampling or p-values"
        ),
    )
    write_json(result, directory / "summary.json")
    return result
