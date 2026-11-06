"""Fresh all-request recovery/control/stability summaries, without a selected winner."""

import numpy as np
from sklearn.metrics import adjusted_rand_score

from super_deli.benchmarking.neural_analysis import arrays, require
from super_deli.benchmarking.neural_summary import summarize_rates
from super_deli.benchmarking.separate_gmm_config import ARMS, HEADS, NATIVE, PRIMARY, tasks
from super_deli.benchmarking.separate_gmm_observer import evaluation_path
from super_deli.benchmarking.separate_gmm_study import check_frozen
from super_deli.benchmarking.view_signal_metrics import linear_recovery
from super_deli.io.config import read_json, write_json


def partition(labels):
    value = labels.copy()
    noise = value < 0
    value[noise] = np.arange(sum(noise)) + max(int(value.max()), 0) + 1
    return value


def finite_mean(value):
    value = np.asarray(value)
    good = value[np.isfinite(value)]
    return float(good.mean()) if len(good) else None


def assess(task, data, arm, head, meta, scores, a):
    row = dict(
        key=task["key"],
        case=task["case"],
        n=task["n"],
        arm=arm + "__" + head,
        status=meta["status"],
        selected_k=meta.get("selected_k"),
        search_boundary=meta.get("selected_k") == 4 if head != "hdbscan" else None,
        endpoint=False,
        endpoint_name="nontrivial_return"
        if task["case"].startswith("gaussian_")
        else "correct_k_full_ari_ge_0_8",
        coverage=None,
        ari=None,
        assigned_ari=None,
        linear_morphology={"r2": [None, None]},
        linear_ecology={"r2": [None, None]},
    )
    if scores is not None:
        for role in ("morphology", "ecology"):
            row["linear_" + role] = linear_recovery(scores, data["latent_" + role])
    if meta["status"] == "available":
        labels = a["labels"]
        assigned = labels >= 0
        partition = labels.copy()
        partition[~assigned] = np.arange(sum(~assigned)) + max(int(labels.max()), 0) + 1
        truth = data["truth"]
        ari = float(adjusted_rand_score(truth, partition))
        row.update(
            coverage=float(assigned.mean()),
            ari=ari,
            assigned_ari=float(adjusted_rand_score(truth[assigned], labels[assigned]))
            if assigned.sum() > 1
            else None,
            endpoint=bool(meta["selected_k"] > 1)
            if task["case"].startswith("gaussian_")
            else bool(meta["selected_k"] == len(set(truth)) and ari >= 0.8),
        )
    return row


def report(root):
    frozen, c = check_frozen(root)
    audit = read_json(root / "main-audit.json")
    require(
        audit["passed"] and (not (root / "STOP.json").exists()), "Audited unmodified main study"
    )
    rows = []
    stability = []
    candidate_rows = []
    for task in tasks(c, "main"):
        folder = root / "main" / task["key"]
        data = arrays(folder / "generating-reference.npz")
        for arm in ARMS:
            out = folder / arm
            scores = None
            p = (
                evaluation_path(out, 0, "coordinates") / "point.npz"
                if arm in NATIVE
                else out / "point.npz"
            )
            if p.exists():
                point = arrays(p)
                if "scores" in point:
                    scores = point["scores"]
                elif all(r + "__scores" in point for r in ("morphology", "ecology")):
                    scores = np.column_stack(
                        [point[r + "__scores"] for r in ("morphology", "ecology")]
                    )
            for head in ("gmm_shared",) if arm == "raw_shared" else HEADS:
                path = evaluation_path(out, 0, head)
                meta = read_json(path / "execution.json")
                a = arrays(path / "fit.npz")
                row = assess(task, data, arm, head, meta, scores, a)
                rows.append(row)
                if head != "hdbscan":
                    candidate_rows.append(
                        dict(
                            key=task["key"],
                            arm=row["arm"],
                            case=task["case"],
                            n=task["n"],
                            status=meta["status"],
                            selected_k=meta.get("selected_k"),
                            candidates=meta.get("candidates", []),
                            attempts=meta.get("candidate_attempts", []),
                        )
                    )
            if arm in NATIVE and task["draws"]:
                for head in HEADS:
                    point_labels = arrays(evaluation_path(out, 0, head) / "fit.npz").get("labels")
                    draws = []
                    ids = np.random.SeedSequence(task["subsampling_path"]).generate_state(1)[0]
                    from super_deli.models.subsample_stability import subsample_indices

                    samples = subsample_indices(task["n"], task["draws"], 0.8, int(ids))
                    for i, sample in enumerate(samples):
                        target = evaluation_path(out, i + 1, head)
                        meta = read_json(target / "execution.json")
                        a = arrays(target / "fit.npz")
                        attempted = (
                            evaluation_path(out, i + 1, "coordinates") / "point.npz"
                        ).exists()
                        draws.append(
                            dict(
                                draw=i,
                                status=meta["status"],
                                pipeline_attempted=attempted,
                                selected_k=meta.get("selected_k"),
                                point_ari=float(
                                    adjusted_rand_score(
                                        partition(point_labels[sample]), partition(a["labels"])
                                    )
                                )
                                if point_labels is not None and meta["status"] == "available"
                                else None,
                            )
                        )
                    from super_deli.models.subsample_stability import summarize_subsamples

                    support_records = []
                    for i, (draw, sample) in enumerate(zip(draws, samples, strict=True)):
                        if not draw["pipeline_attempted"]:
                            continue
                        record = dict(
                            draw=i,
                            training_indices=sample.tolist(),
                            status="complete" if draw["status"] == "available" else "failed",
                        )
                        if draw["status"] == "available":
                            saved = arrays(evaluation_path(out, i + 1, head) / "fit.npz")
                            record["labels"] = saved["labels"].tolist()
                        support_records.append(record)
                    support = summarize_subsamples(task["n"], support_records)
                    support_dir = root / "report-support"
                    support_dir.mkdir(exist_ok=True)
                    relative = "report-support/" + task["key"] + "__" + arm + "__" + head + ".npz"
                    np.savez_compressed(root / relative, **support.arrays)
                    stability.append(
                        dict(
                            key=task["key"],
                            case=task["case"],
                            n=task["n"],
                            arm=arm + "__" + head,
                            draws=draws,
                            requested=task["draws"],
                            support=support.summary,
                            support_path=relative,
                            mean_coassignment=finite_mean(support.arrays.get("coassignment", [])),
                            mean_coassignment_lower=finite_mean(
                                support.arrays.get("coassignment_lower", [])
                            ),
                            mean_coassignment_upper=finite_mean(
                                support.arrays.get("coassignment_upper", [])
                            ),
                        )
                    )
    arms = [a + "__" + h for a in ARMS for h in (("gmm_shared",) if a == "raw_shared" else HEADS)]
    joint = [a + "__" + h for a in NATIVE for h in HEADS]
    comparison = [
        dict(
            reference=a + "__" + h,
            alternative=a + "__gmm_shared",
            labels=dict(reference_arm=a + "__" + h, comparison_arm=a + "__gmm_shared"),
            positive_key="shared_only",
            negative_key="reference_only",
        )
        for a in NATIVE
        for h in ("hdbscan", "gmm_full")
    ]
    groups, primary, paired = summarize_rates(
        rows, arms=arms, joint=joint, comparisons=comparison, primary_cases=PRIMARY
    )
    for group in groups:
        group.pop("standardized_mse_by_view", None)
    audits = {}
    for phase in ("preflight", "main"):
        value = read_json(root / (phase + "-audit.json"))
        audits[phase] = {k: v for k, v in value.items() if k != "records"}
        for field in ("candidate_densities_checked", "density_fits"):
            audits[phase][field] = sum(r[field] for r in value["records"])
    summary = dict(
        schema_version=1,
        fixture=c.fixture,
        observations=len(tasks(c, "main")),
        counts=frozen["counts"],
        primary_rates=len(primary),
        paired_comparisons=len(paired),
        audits=audits,
        defaults_changed=False,
        historical_pvalues_changed=False,
        new_pvalues=0,
        interpretation=(
            "Fresh prespecified simulations; seeds/heads repeat observations; "
            "no selected winner, species or general size guarantee"
        ),
    )
    for name, value in (
        ("assessments", rows),
        ("all-groups", groups),
        ("primary-rates", primary),
        ("paired-comparisons", paired),
        ("candidate-assessments", candidate_rows),
        ("native-stability", stability),
        ("summary", summary),
    ):
        write_json(value, root / (name + ".json"))
    return summary
