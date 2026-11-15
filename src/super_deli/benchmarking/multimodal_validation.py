"""Read-only numerical observers and independent shape/native-output audits."""

from contextlib import ExitStack, contextmanager
from hashlib import sha256
from unittest.mock import patch


def array_hash(values):
    import numpy as np

    return sha256(np.asarray(values, dtype="<f8").tobytes(order="C")).hexdigest()


@contextmanager
def observe_shape_refits():
    import numpy as np

    from super_deli.models import integrative, shape_gmm, shape_pca

    records = []

    def wrapper(native, owner):
        def checked(raw, *, config=None):
            x = np.asarray(raw, dtype=float)
            record = dict(
                owner=owner, raw_shape=list(x.shape), raw_sha256=array_hash(x), status="failed"
            )
            records.append(record)
            fitted = native(raw, config=config)
            model, projection = fitted.model, fitted.projection
            centered = x - x.mean(axis=1, keepdims=True)
            sizes = np.sqrt((centered * centered).sum(axis=(1, 2)))
            y = centered / sizes[:, None, None]
            consensus = model.consensus
            np.testing.assert_allclose(consensus.mean(axis=0), 0, atol=1e-9)
            np.testing.assert_allclose(np.linalg.norm(consensus), 1, rtol=1e-7, atol=1e-9)
            independent = []
            for unit in y:
                u, _, vt = np.linalg.svd(unit.T @ consensus)
                if model.config.reflection == "forbid" and np.linalg.det(u @ vt) < 0:
                    u[:, -1] *= -1
                independent.append(unit @ u @ vt)
            aligned = np.asarray(independent)
            np.testing.assert_allclose(aligned, projection.aligned, rtol=1e-7, atol=1e-9)
            cosine = np.einsum("nld,ld->n", aligned, consensus)
            tangent = aligned - cosine[:, None, None] * consensus
            scores = (tangent.reshape(len(x), -1) - model.tangent_mean) @ model.components.T
            np.testing.assert_allclose(scores, projection.scores, rtol=1e-7, atol=1e-9)
            np.testing.assert_allclose(
                model.components @ model.components.T,
                np.eye(len(model.components)),
                rtol=1e-7,
                atol=1e-9,
            )
            record.update(
                status="complete",
                iterations=len(fitted.history),
                numerical_rank=model.numerical_rank,
                configuration=model.config.to_dict(),
            )
            return fitted

        return checked

    # Import every alias before patching; nested observations must not count twice.
    originals = [
        (shape_pca, "shape_point", shape_pca.fit_shape_pca),
        (shape_gmm, "shape_bootstrap", shape_gmm.fit_shape_pca),
        (integrative, "integrative", integrative.fit_shape_pca),
    ]
    with ExitStack() as stack:
        for module, owner, native in originals:
            stack.enter_context(patch.object(module, "fit_shape_pca", wrapper(native, owner)))
        yield records


def equal_point_arrays(left, right):
    import numpy as np

    count = 0
    for path in sorted(left.rglob("*")):
        if path.suffix not in {".npz", ".npy"} or path.name.startswith(
            ("stability", "coassignment")
        ):
            continue
        other = right / path.relative_to(left)
        assert other.is_file(), path
        if path.suffix == ".npy":
            np.testing.assert_array_equal(
                np.load(path, allow_pickle=False), np.load(other, allow_pickle=False)
            )
            count += 1
        else:
            with np.load(path, allow_pickle=False) as a, np.load(other, allow_pickle=False) as b:
                assert a.files == b.files
                for key in a.files:
                    np.testing.assert_array_equal(a[key], b[key])
                    count += 1
    return count


def selected_density(directory, arm):
    import numpy as np
    from scipy.special import logsumexp

    from super_deli.io.config import read_json

    summary = read_json(directory / "summary.json")
    with np.load(directory / "mixture.npz", allow_pickle=False) as model:
        w, means = model["weights"], model["means"]
        if arm.family == "genomic":
            scores = np.load(directory / "pca_scores.npy", allow_pickle=False)
            membership = np.load(directory / "membership.npy", allow_pickle=False)
            covariances = np.asarray([np.diag(v) for v in model["variances"]])
        else:
            scores, membership, covariances = (
                model["scores"],
                model["membership"],
                model["covariances"],
            )
    n, d = scores.shape
    k = len(w)
    terms = []
    for weight, mean, cov in zip(w, means, covariances, strict=True):
        sign, logdet = np.linalg.slogdet(cov)
        assert sign == 1 and np.linalg.eigvalsh(cov).min() > 0
        residual = scores - mean
        terms.append(
            np.log(weight)
            - 0.5
            * (
                d * np.log(2 * np.pi)
                + logdet
                + np.einsum("ij,ji->i", residual, np.linalg.solve(cov, residual.T))
            )
        )
    terms = np.asarray(terms).T
    total = logsumexp(terms, axis=1)
    np.testing.assert_allclose(np.exp(terms - total[:, None]), membership, rtol=1e-7, atol=1e-9)
    covariance_parameters = (
        k * d
        if arm.family == "genomic"
        else d
        * (d + 1)
        // 2
        * (1 if arm.family == "integrative" and arm.model.covariance_type == "tied" else k)
    )
    bic = float(-2 * total.sum() + (k - 1 + k * d + covariance_parameters) * np.log(n))
    candidates = (
        summary["candidates"]
        if arm.family == "genomic"
        else read_json(directory / "candidates.json")
    )
    selected = next(r for r in candidates if r["k"] == k)
    np.testing.assert_allclose(bic, selected["bic"], rtol=1e-8, atol=1e-7)
    assert selected["status"] == "eligible" and selected["bic"] == min(
        r["bic"] for r in candidates if r["status"] == "eligible"
    )
    return dict(passed=True, selected_k=k, bic=bic)
