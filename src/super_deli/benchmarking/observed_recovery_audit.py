"""Independent reconstruction of frozen laws, continuations, guards and oracle isolation."""

import hashlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from super_deli.benchmarking.observed_audit import (
    audit_fit,
    audit_observation,
    change,
    reference,
    reference_update,
)
from super_deli.io.config import read_json, write_json

KEYS = ("weights", "means", "covariances")


def arrays(path):
    import numpy as np

    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def independent_inputs(task):
    """No calls to production or study generators/masking functions."""
    import numpy as np

    c = ["spherical", "correlated", "four_groups"].index(task["case"])
    ns = [2 if task["phase"] == "preflight" else 3, c, task["n"], task["replicate"]]
    assert task["namespace"] == ns

    def generate(n, seed):
        rng = np.random.default_rng(seed)
        if c == 0:
            return rng.normal(size=(n, 4)), np.zeros(n, dtype=int)
        rho = 0.6 if c == 1 else 0.3
        covariance = np.fromfunction(lambda i, j: rho ** abs(i - j), (4, 4))
        if c == 1:
            return rng.multivariate_normal(np.zeros(4), covariance, n), np.zeros(n, dtype=int)
        truth = rng.integers(0, 4, size=n)
        left, right = 2 * (truth // 2) - 1, 2 * (truth % 2) - 1
        means = np.stack([3 * left, 3 * right, 1.5 * left, 1.5 * right], axis=1)
        return means + rng.multivariate_normal(np.zeros(4), covariance, n), truth

    def missing(x, seed):
        uniform = np.random.default_rng(seed + [0]).random(x.shape)
        mar = np.random.default_rng(seed + [1]).random(x.shape) < np.where(
            x[:, :1] <= 0, 0.15, 0.65
        )
        mar[:, 0] = False
        return dict(mcar30=uniform < 0.3, mar=mar)

    x, truth = generate(task["n"], [81403] + ns)
    evaluation, ev_truth = generate(2048, [81429] + ns + [0])
    return dict(
        complete=x,
        truth=truth,
        evaluation=evaluation,
        evaluation_truth=ev_truth,
        **{f"mask_{k}": v for k, v in missing(x, [81409] + ns).items()},
        **{f"eval_mask_{k}": v for k, v in missing(evaluation, [81429] + ns + [1]).items()},
    )


def assert_replay(source, replay):
    import numpy as np

    a, b = read_json(source / "summary.json"), read_json(replay / "summary.json")
    for summary in (a, b):
        summary.pop("reference_common_n", None)
        summary.pop("reference_common_ari", None)
    assert a == b
    for filename in ("fit.npz", "rejected-input.npz", "evaluation.npz"):
        if not (source / filename).exists():
            assert not (replay / filename).exists()
            continue
        first, second = arrays(source / filename), arrays(replay / filename)
        assert set(first) == set(second)
        for key in first:
            np.testing.assert_array_equal(first[key], second[key])


def audit_continuation(source, target):
    import numpy as np

    original, new = read_json(source / "summary.json"), read_json(target / "summary.json")
    a, b = arrays(source / "fit.npz"), arrays(target / "fit.npz")
    assert new["baseline_sha256"] == {
        name: file_hash(source / name) for name in ("summary.json", "fit.npz")
    }
    assert new["config"] == original["config"]
    assert new["total_iteration_cap"] == 10000
    assert new["original_result_replaced"] is False and new["oracle_used"] is False
    for name in ("raw", "observed", "coobserved", "kept_indices", "normalization_means", "scales"):
        np.testing.assert_array_equal(a[name], b[name])
    x = (a["raw"] - a["normalization_means"]) / a["scales"]
    attempts = 0
    for old, updated in zip(original["candidates"], new["candidates"], strict=True):
        for first, second in zip(old["starts"], updated["starts"], strict=True):
            prefix = f"k{old['k']}_s{first['start']}_"
            continued = (
                first["status"] == "failed"
                and first.get("error") == "Iteration limit reached"
                and old["k"] in new.get("continued_components", list(range(1, 5)))
            )
            if not continued:
                assert first == second
                for key in a:
                    if key.startswith(prefix):
                        np.testing.assert_array_equal(a[key], b[key])
                continue
            attempts += 1
            assert first["iterations"] == 2000
            assert second["original_iterations"] == 2000
            assert second["iterations"] == 2000 + second["additional_iterations"] <= 10000
            assert second["original_status"] == first["status"]
            assert second["original_error"] == first["error"]
            np.testing.assert_array_equal(b[prefix + "trace"][:2001], a[prefix + "trace"])
            anchor = tuple(b[prefix + "anchor_" + name] for name in KEYS)
            for name, value in zip(KEYS, anchor, strict=True):
                np.testing.assert_array_equal(value, a[prefix + "final_" + name])
                np.testing.assert_array_equal(
                    b[prefix + "initial_" + name], a[prefix + "initial_" + name]
                )
            first_update = reference_update(x, anchor, 1e-4)
            if second["additional_iterations"]:
                np.testing.assert_allclose(
                    reference(x, *first_update)[0].sum(), b[prefix + "trace"][2001], atol=3e-6
                )
                previous = tuple(b[prefix + "previous_" + name] for name in KEYS)
                final = tuple(b[prefix + "final_" + name] for name in KEYS)
                for expected, value in zip(reference_update(x, previous, 1e-4), final, strict=True):
                    np.testing.assert_allclose(expected, value, atol=3e-7, rtol=2e-7)
    assert attempts == new["continuation_attempts"]
    return attempts


def audit_oracle(source, target):
    import numpy as np

    base, result = read_json(source / "summary.json"), read_json(target / "summary.json")
    a, b = arrays(source / "fit.npz"), arrays(target / "fit.npz")
    assert base["task"]["case"] == "four_groups"
    assert result["excluded_from_selection"] is True and "selected_k" not in result
    assert result["total_iteration_cap"] == 10000 and result["iterations"] <= 10000
    assert result["baseline_sha256"] == {
        name: file_hash(source / name) for name in ("summary.json", "fit.npz")
    }
    for name in ("raw", "normalization_means", "scales", "kept_indices"):
        np.testing.assert_array_equal(a[name], b[name])
    means = np.array(
        [[-3, -3, -1.5, -1.5], [-3, 3, -1.5, 1.5], [3, -3, 1.5, -1.5], [3, 3, 1.5, 1.5]]
    )
    scales = a["scales"]
    covariance = np.fromfunction(lambda i, j: 0.3 ** abs(i - j), (4, 4)) / np.outer(scales, scales)
    eigen, vectors = np.linalg.eigh(covariance)
    bounded = (vectors * np.maximum(eigen, 1e-4)) @ vectors.T
    np.testing.assert_allclose(b["initial_weights"], 0.25, rtol=0, atol=0)
    np.testing.assert_allclose(b["initial_means"], (means - a["normalization_means"]) / scales)
    np.testing.assert_allclose(b["initial_covariances"], np.repeat(bounded[None], 4, axis=0))
    x = (a["raw"] - a["normalization_means"]) / scales
    trace = b["trace"]
    assert len(trace) == result["iterations"] + 1 and np.isfinite(trace).all()
    assert np.all(
        np.diff(trace)
        >= -np.maximum(1e-8, 128 * np.finfo(float).eps * np.maximum(1, abs(trace[:-1])))
    )
    for stage in ("initial", "final"):
        p = tuple(b[stage + "_" + name] for name in KEYS)
        assert np.linalg.eigvalsh(p[2]).min() >= 1e-4 * (1 - 1e-7)
        np.testing.assert_allclose(
            reference(x, *p)[0].sum(), trace[0 if stage == "initial" else -1], atol=3e-6
        )
    if result["iterations"]:
        previous = tuple(b["previous_" + name] for name in KEYS)
        final = tuple(b["final_" + name] for name in KEYS)
        for expected, value in zip(reference_update(x, previous, 1e-4), final, strict=True):
            np.testing.assert_allclose(expected, value, atol=3e-7, rtol=2e-7)
    if result["status"] == "converged":
        assert abs(trace[-1] - trace[-2]) / len(x) <= 1e-7 + 1e-10
        assert change(final, reference_update(x, final, 1e-4)) <= 1e-5 + 2e-8
        _, membership = reference(x, *final)
        np.testing.assert_allclose(b["membership"], membership, atol=2e-8)
        sizes = np.bincount(membership.argmax(axis=1), minlength=4)
        assert result["hard_cluster_sizes"] == sizes.tolist()
        assert result["eligible"] == bool(sizes.min() >= 5)
    return 1


def audit_new_observation(directory):
    import numpy as np

    directory = Path(directory)
    task = read_json(directory / "task.json")
    expected = independent_inputs(task)
    stored = arrays(directory / "inputs.npz")
    assert set(stored) == set(expected)
    for name in expected:
        np.testing.assert_array_equal(stored[name], expected[name])
    count = searches = starts = oracles = 0
    for mask in ("mcar30", "mar"):
        masked = np.where(expected[f"mask_{mask}"], np.nan, expected["complete"])
        mean = np.nanmean(masked, axis=0)
        scale = np.sqrt(np.nanmean((masked - mean) ** 2, axis=0))
        for arm in ("observed", "reference"):
            condition = directory / f"{mask}_{arm}"
            values = masked if arm == "observed" else expected["complete"]
            arguments = (
                values,
                np.arange(task["n"]),
                mean,
                scale,
                expected["evaluation"],
                expected[f"eval_mask_{mask}"],
                expected["truth"],
            )
            base = audit_fit(condition / "baseline", *arguments)
            starts += base["starts"]
            searches += 1
            if (condition / "extended").exists():
                audit_fit(condition / "extended", *arguments)
                count += audit_continuation(condition / "baseline", condition / "extended")
            else:
                assert base["status"] == "rejected_information_guard"
            if task["case"] == "four_groups":
                oracles += audit_oracle(condition / "baseline", condition / "oracle")
            if (condition / "replay").exists():
                repeated = audit_fit(condition / "replay", *arguments)
                starts += repeated["starts"]
                assert_replay(condition / "baseline", condition / "replay")
                searches += 1
    return dict(
        key=task["key"], searches=searches, starts=starts, continuations=count, oracles=oracles
    )


def archive_audit(root, workers=4):
    root = Path(root)
    archive = Path(read_json(root / "frozen-before-generation.json")["archive"])
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(audit_observation, sorted((archive / "main").iterdir())))
    assert sum(r["searches"] for r in results) == 1224
    assert sum(r["point_failures"] for r in results) == 4
    rejected = archive / "main/spherical_n64_08/pattern_complete_case"
    summary = read_json(rejected / "summary.json")
    data = arrays(rejected / "rejected-input.npz")["raw"]
    assert len(data) == 4 and len(data) < max(5, data.shape[1] + 2)
    assert summary["error"] == "Too few informative independent units"
    result = dict(
        status="passed",
        searches=1224,
        starts=sum(r["starts"] for r in results),
        point_failures=4,
        information_rejection_preserved=True,
        original_files_written=False,
        observations=results,
    )
    write_json(result, root / "archive-audit.json")
    return result


def audit_archived_action(arguments):
    import numpy as np

    archive, folder, action = arguments
    key, condition = folder.name.split("__", 1)
    source = archive / "main" / key / condition
    data = arrays(archive / "main" / key / "inputs.npz")
    mask, arm = condition.rsplit("_", 1)
    original = read_json(source / "summary.json")
    a = arrays(source / "fit.npz")
    values = (
        np.where(data[f"mask_{mask}"], np.nan, data["complete"])
        if arm == "observed"
        else data["complete"]
    )
    if action == "oracle":
        return dict(oracles=audit_oracle(source, folder))
    checked = audit_fit(
        folder,
        values,
        np.arange(len(values)),
        a["normalization_means"],
        a["scales"],
        data["evaluation"],
        data[f"eval_mask_{mask}"],
        data["truth"],
    )
    if action == "replay":
        assert_replay(source, folder)
        return dict(searches=1, starts=checked["starts"])
    assert original["status"] == "failed"
    attempts = audit_continuation(source, folder)
    assert attempts == 10
    assert read_json(folder / "summary.json")["continued_components"] == [4]
    return dict(continuations=attempts)


def audit_phase(root, phase, workers=4):
    root = Path(root)
    folders = sorted((root / phase).iterdir())
    assert len(folders) == (6 if phase == "preflight" else 60)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(audit_new_observation, folders))
        archived = []
        if phase == "main":
            archive = Path(read_json(root / "frozen-before-generation.json")["archive"])
            jobs = [
                (archive, p, action)
                for action, plural in (
                    ("continuation", "continuations"),
                    ("oracle", "oracles"),
                    ("replay", "replays"),
                )
                for p in sorted((root / f"archived-{plural}").iterdir())
            ]
            archived = list(executor.map(audit_archived_action, jobs))
    counts = {
        key: sum(r.get(key, 0) for r in results + archived)
        for key in ("searches", "starts", "continuations", "oracles")
    }
    assert counts["searches"] == (30 if phase == "preflight" else 252)
    assert counts["oracles"] == (8 if phase == "preflight" else 160)
    assert counts["continuations"] <= (960 if phase == "preflight" else 9630)
    excluded = 0
    if phase == "main" and (root / "orchestration-correction.json").exists():
        correction = read_json(root / "orchestration-correction.json")
        excluded = correction["unplanned_continuations_excluded"]
        assert excluded == 24 and correction["software_fits_rerun"] == 0
        for name, expected in correction["retained_evidence_sha256"].items():
            assert file_hash(root / name) == expected
    result = dict(
        status="passed",
        phase=phase,
        unplanned_continuations_excluded=excluded,
        **counts,
        observations=results,
        archived_actions=archived,
        production_EM_or_selection_called=False,
    )
    write_json(result, root / f"{phase}-audit.json")
    return result


def report(root):
    from super_deli.benchmarking.observed_recovery_report import build_report

    return build_report(root)
