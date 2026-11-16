"""Independent transforms, density/eligibility and untouched-cohort predictions."""

from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist
from scipy.special import logsumexp

from super_deli.benchmarking.full_retention_audit import _search, _shape, _view, close
from super_deli.benchmarking.multimodal_validation import array_hash
from super_deli.benchmarking.signal_loss_metrics import parameters
from super_deli.io.config import read_json, write_json


class NumericalAuditFailure(ValueError):
    """Withhold a diagnostic search; never replace its result with K=1."""


def logdensity(x, a):
    terms = []
    for w, m in zip(a["weights"], a["means"], strict=True):
        c = a["covariances"]
        sign, ld = np.linalg.slogdet(c)
        if sign != 1 or np.linalg.eigvalsh(c).min() <= 0:
            raise NumericalAuditFailure("Nonpositive covariance")
        z = x - m
        terms.append(
            np.log(w)
            - 0.5
            * (
                x.shape[1] * np.log(2 * np.pi)
                + ld
                + np.einsum("ij,ji->i", z, np.linalg.solve(c, z.T))
            )
        )
    terms = np.array(terms).T
    total = logsumexp(terms, axis=1)
    return total, np.exp(terms - total[:, None]), terms.argmax(1)


def external_scores(inputs, units, event, a):
    blocks = []
    for i, (data, view) in enumerate(zip(inputs, event["views"], strict=True)):
        p = f"v{i}_"
        lookup = {u: j for j, u in enumerate(data.target_ids)}
        if view["kind"] == "morphometric":
            raw = _shape(
                data.values, a[p + "consensus"], view["configuration"]["shape"]["reflection"]
            )
        else:
            raw = (data.values[:, a[p + "kept"]] - a[p + "feature_means"]) / a[p + "feature_scales"]
        pc = (raw - a[p + "pca_mean"]) @ a[p + "components"].T
        blocks.append(
            ((pc - a[p + "mean"]) * np.sqrt(a[p + "weight"]) / a[p + "scale"])[
                [lookup[u] for u in units]
            ]
        )
    return (np.concatenate(blocks, axis=1) - a["pca_mean"]) @ a["components"].T


def audit(
    directory,
    original,
    evaluation,
    *,
    head=False,
    candidate_directory="candidates",
    pipeline_directory="pipeline",
    write=True,
):
    directory = Path(directory)
    records = read_json(directory / candidate_directory / "index.json")
    events = read_json(
        directory / "events.json" if head else directory / pipeline_directory / "events.json"
    )
    predictions = iter(read_json(directory / "predictions.json"))
    rows = []
    checked = 0
    layout = None if head else read_json(directory / pipeline_directory / "raw-layout.json")
    raw = (
        []
        if head
        else [
            np.load(directory / pipeline_directory / f"raw_{i}.npy", allow_pickle=False)
            for i in range(len(layout["views"]))
        ]
    )
    for event in events:
        if event["status"] != "complete":
            rows.append(dict(status="withheld_fit", error=event.get("error")))
            continue
        prediction = next(predictions)
        assert (
            event["selected_k"] == prediction["selected_k"]
            and event["candidates"] == prediction["candidates"]
        )
        fits = records[event["candidate_start"] : event["candidate_stop"]]
        with np.load(directory / prediction["arrays"], allow_pickle=False) as loaded:
            a = {k: loaded[k].copy() for k in loaded.files}
        if head:
            with np.load(directory / "head-inputs.npz", allow_pickle=False) as f:
                x = f["train"][event["draw"]]
            np.testing.assert_array_equal(a["original_scores"], original)
            np.testing.assert_array_equal(a["evaluation_scores"], evaluation)
        else:
            with np.load(directory / pipeline_directory / event["arrays"], allow_pickle=False) as f:
                b = {k: f[k].copy() for k in f.files}
            blocks = [
                _view(
                    raw[i], v, b, f"v{i}_", layout["views"][i], layout["unit_ids"], event["draw"]
                )[0]
                for i, v in enumerate(event["views"])
            ]
            fused = np.concatenate(blocks, axis=1)
            close(fused, b["fused"])
            close((fused - b["pca_mean"]) @ b["components"].T, b["scores"])
            close(pdist(fused), pdist(b["scores"]))
            close(external_scores(*original, event, b), a["original_scores"])
            close(external_scores(*evaluation, event, b), a["evaluation_scores"])
            x = b["scores"]
        for r in fits:
            assert r["n_init"] == 100 and r["scores_sha256"] == array_hash(x)
            assert r["seed"] == int(
                np.random.SeedSequence([event["seed"], r["k"]]).generate_state(1)[0]
            )
        gains = {}
        failure = None
        try:
            for r in fits:
                assert r["fit_status"] == "complete"
                with np.load(
                    directory / candidate_directory / r["arrays"], allow_pickle=False
                ) as loaded:
                    c = {k: loaded[k].copy() for k in loaded.files}
                np.testing.assert_array_equal(c["scores"], x)
                ll, prob, labels = logdensity(x, c)
                np.testing.assert_allclose(prob, c["membership"], rtol=1e-7, atol=1e-9)
                np.testing.assert_array_equal(c["labels"], c["membership"].argmax(1))
                np.testing.assert_array_equal(labels, c["labels"])
                np.testing.assert_allclose(ll.sum(), r["log_likelihood"], rtol=1e-8, atol=1e-7)
                np.testing.assert_allclose(
                    -2 * ll.sum() + parameters(r["k"], 8, "tied") * np.log(len(x)),
                    r["bic"],
                    rtol=1e-8,
                    atol=1e-7,
                )
                el, ep, ey = logdensity(a["evaluation_scores"], c)
                np.testing.assert_allclose(
                    el, a[f"k{r['k']}_evaluation_logdensity"], rtol=1e-8, atol=1e-7
                )
                np.testing.assert_allclose(
                    ep, a[f"k{r['k']}_evaluation_membership"], rtol=1e-7, atol=1e-9
                )
                if r["k"] == event["selected_k"]:
                    np.testing.assert_array_equal(ey, a["evaluation_labels"])
                    ol, op, oy = logdensity(a["original_scores"], c)
                    np.testing.assert_allclose(op, a["original_membership"], rtol=1e-7, atol=1e-9)
                    np.testing.assert_array_equal(oy, a["original_labels"])
                gains[r["k"]] = float(el.mean())
                checked += 1
        except (AssertionError, NumericalAuditFailure) as exc:
            failure = f"{type(exc).__name__}: {exc}"
        if failure:
            rows.append(
                dict(status="withheld_numerical_audit", error=failure, index=event["index"])
            )
            continue
        _search(event, records, directory / candidate_directory, event["draw"])
        rows.append(
            dict(
                status="complete",
                index=event["index"],
                selected_k=event["selected_k"],
                evaluation_gain=gains[event["selected_k"]] - gains[1],
                candidate_evaluation_gains={str(k): v - gains[1] for k, v in gains.items()},
                candidate_costs={
                    str(r["k"]): dict(
                        twice_likelihood_gain=2 * (r["log_likelihood"] - fits[0]["log_likelihood"]),
                        extra_bic_penalty=(parameters(r["k"], 8, "tied") - parameters(1, 8, "tied"))
                        * np.log(len(x)),
                        bic_gain=fits[0]["bic"] - r["bic"],
                    )
                    for r in fits
                },
                score_covariance_eigenvalues=np.linalg.eigvalsh(np.cov(x, rowvar=False)).tolist(),
                unique_originals=len(set(event["draw"])),
                training_rows=len(event["draw"]),
                candidates=event["candidates"],
                prediction_arrays=prediction["arrays"],
            )
        )
    assert next(predictions, None) is None
    result = dict(
        passed=True,
        rows=rows,
        density_checks=checked,
        attempted_searches=len(events),
        withheld=sum(r["status"] != "complete" for r in rows),
    )
    if write:
        write_json(result, directory / "diagnostic-audit.json")
    return result
