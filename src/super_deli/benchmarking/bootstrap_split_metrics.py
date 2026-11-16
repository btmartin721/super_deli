"""Parent-level diagnostic denominators and independently reconstructed bootstrap counts."""

from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score

from super_deli.io.config import read_json


def native_counts(directory, n, seed, draws):
    directory = Path(directory)
    events = read_json(directory / "pipeline/events.json")
    if not (directory / "native-bootstrap.json").exists():
        return dict(point_failed=True)
    reps = read_json(directory / "native-bootstrap.json")
    assert len(events) == draws + 1 and len(reps) == draws
    children = np.random.SeedSequence(seed).spawn(draws + 1)
    assert events[0]["seed"] == int(children[0].generate_state(1)[0])
    np.testing.assert_array_equal(events[0]["draw"], np.arange(n))
    co, oc, rq, ok = [np.zeros((n, n), dtype=np.uint64) for _ in range(4)]
    for event, rep, child in zip(events[1:], reps, children[1:], strict=True):
        ds, fs = map(int, child.generate_state(2))
        draw = np.random.default_rng(ds).integers(n, size=n)
        assert rep["draw_seed"] == ds and rep["fit_seed"] == fs == event["seed"]
        np.testing.assert_array_equal(draw, rep["training_indices"])
        np.testing.assert_array_equal(draw, event["draw"])
        absent = np.bincount(draw, minlength=n) == 0
        pairs = np.logical_and.outer(absent, absent)
        rq += pairs
        assert rep["oob_indices"] == np.flatnonzero(absent).tolist() and rep[
            "unique_training_units"
        ] == len(set(draw))
        assert rep["status"] == event["status"]
        if event["status"] == "complete":
            with np.load(directory / "pipeline" / event["arrays"], allow_pickle=False) as a:
                np.testing.assert_array_equal(rep["labels"], a["projected_labels"])
            same = np.equal.outer(rep["labels"], rep["labels"])
            co += same
            oc += same & pairs
            ok += pairs
    with np.load(directory / "native-stability.npz", allow_pickle=False) as a:
        for key, value in (
            ("coassignment_counts", co),
            ("oob_coassignment_counts", oc),
            ("oob_requested", rq),
            ("oob_successful", ok),
        ):
            np.testing.assert_array_equal(a[key], value)
        failures = sum(r["status"] != "complete" for r in reps)
        lower = co / draws
        upper = (co + failures) / draws
        np.fill_diagonal(lower, 1)
        np.fill_diagonal(upper, 1)
        np.testing.assert_allclose(a["coassignment_lower"], lower)
        np.testing.assert_allclose(a["coassignment_upper"], upper)
        np.testing.assert_allclose(
            a["coassignment"], lower if not failures else np.full((n, n), np.nan)
        )
        low = np.full((n, n), np.nan)
        high = low.copy()
        np.divide(oc, rq, out=low, where=rq > 0)
        np.divide(oc + rq - ok, rq, out=high, where=rq > 0)
        np.testing.assert_allclose(a["oob_coassignment_lower"], low)
        np.testing.assert_allclose(a["oob_coassignment_upper"], high)
        np.testing.assert_allclose(a["oob_coassignment"], np.where(rq == ok, low, np.nan))
    return dict(passed=True, draws=draws, point_failed=False)


def measure(directory, rows, truth, point, draws=None):
    directory = Path(directory)
    out = []
    for i, r in enumerate(rows):
        row = {k: v for k, v in r.items() if k not in {"candidates", "error"}}
        if r["status"] != "complete":
            out.append(row)
            continue
        with np.load(directory / r["prediction_arrays"], allow_pickle=False) as a:
            labels = a["original_labels"].copy()
        row.update(
            ari=float(adjusted_rand_score(truth, labels)),
            point_ari=None if point is None else float(adjusted_rand_score(point, labels)),
        )
        if point is not None:
            pairs = np.triu_indices(len(labels), 1)
            row["common_point_pair_agreement"] = float(
                np.mean(
                    (labels[:, None] == labels[None, :])[pairs]
                    == (point[:, None] == point[None, :])[pairs]
                )
            )
            groups = [set(np.flatnonzero(point == j)) for j in np.unique(point)]
            parts = [set(np.flatnonzero(labels == j)) for j in np.unique(labels)]
            row["exact_point_clusters"] = sum(g in parts for g in groups)
            row["merged_point_clusters"] = sum(any(g < h for h in parts) for g in groups)
        if draws is not None:
            frequency = np.bincount(draws[i], minlength=len(labels)) / len(draws[i])
            row["effective_weight_support"] = float(1 / np.sum(frequency**2))
            absent = np.bincount(draws[i], minlength=len(labels)) == 0
            row.update(
                oob_units=int(absent.sum()),
                oob_ari=float(adjusted_rand_score(truth[absent], labels[absent]))
                if absent.sum() > 1
                else None,
            )
        k4 = next(c for c in r["candidates"] if c["k"] == 4)
        k1 = next(c for c in r["candidates"] if c["k"] == 1)
        row.update(
            k4_status=k4["status"],
            k4_distinct_sizes=k4.get("distinct_unit_sizes"),
            k4_bic_gain=None if k4.get("bic") is None else k1["bic"] - k4["bic"],
        )
        out.append(row)
    return out
