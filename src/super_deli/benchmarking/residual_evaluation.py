"""Frozen external-cohort projection and task-specific descriptive metrics."""

from dataclasses import replace

import numpy as np
from scipy.special import logsumexp
from sklearn.metrics import adjusted_rand_score, confusion_matrix

from super_deli.models.matched_preprocessing import align_tables


def aligned_matrix(tables, config):
    ids = sorted(tables[0].unit_ids)
    if len(set(ids)) != len(ids) or any(set(t.unit_ids) != set(ids) for t in tables):
        raise ValueError("Complete unique external cohort identities required")
    blocks = []
    for start in range(0, len(ids), 512):
        wanted = ids[start : start + 512]
        subsets = []
        for table in tables:
            positions = {u: i for i, u in enumerate(table.unit_ids)}
            rows = [positions[u] for u in wanted]
            subsets.append(
                replace(
                    table,
                    observation_ids=tuple(table.observation_ids[i] for i in rows),
                    unit_ids=tuple(wanted),
                    values=tuple(table.values[i] for i in rows),
                )
            )
        x, manifest = align_tables(subsets, config, minimum=1)
        if manifest["unit_ids"] != wanted:
            raise ValueError("Projection unit order changed")
        blocks.append(x)
    return np.vstack(blocks), ids


def encode_matrix(x, arrays, arm):
    from super_deli.models.tabular_neural import forward_numpy

    parts = []
    for role, view in (("morphology", x[:, :4]), ("ecology", x[:, 4:])):
        a = {k.split("__", 1)[1]: v for k, v in arrays.items() if k.startswith(role + "__")}
        kept = a["kept_indices"]
        z = (view[:, kept] - a["means"][kept]) / a["scales"][kept]
        raw = (
            (z - a["pca_mean"]) @ a["components"].T
            if arm == "pca4"
            else forward_numpy(z, a, "encoder")
        )
        parts.append((raw - a["latent_means"]) / a["latent_scales"])
    return np.column_stack(parts)


def frozen_membership(scores, head, metadata):
    k = metadata["selected_k"]
    order = metadata["component_order"]
    means, weights = head[f"k{k}_means"][order], head[f"k{k}_weights"][order]
    covariance = head[f"k{k}_covariances"]
    factor = np.linalg.cholesky(covariance)
    difference = scores[:, None, :] - means[None, :, :]
    residual = np.linalg.solve(factor, difference.reshape(-1, 4).T).T.reshape(difference.shape)
    logp = np.log(weights) - 0.5 * (
        4 * np.log(2 * np.pi)
        + 2 * np.log(np.diag(factor)).sum()
        + np.sum(residual * residual, axis=2)
    )
    return np.exp(logp - logsumexp(logp, axis=1)[:, None])


def classification_metrics(y, p):
    if p.shape != (len(y), 4) or not np.isfinite(p).all() or np.any(p < 0):
        raise ValueError("Four-class probabilities required for known-class evaluation")
    np.testing.assert_allclose(p.sum(1), 1, rtol=1e-10, atol=1e-10)
    chosen = p.argmax(1)
    confusion = confusion_matrix(y, chosen, labels=np.arange(4))
    accuracy = np.diag(confusion) / confusion.sum(1)
    target = np.eye(4)[y]
    return dict(
        balanced_accuracy=float(accuracy.mean()),
        confusion=confusion.tolist(),
        log_loss=float(-np.mean(np.log(np.maximum(p[np.arange(len(y)), y], np.finfo(float).tiny)))),
        brier=float(np.mean(np.sum((p - target) ** 2, axis=1))),
    )


def confidence_metrics(p):
    maximum = p.max(1)
    return dict(
        n=len(p),
        mean_maximum_reference_confidence=float(maximum.mean()),
        fraction_above_point_nine=float(np.mean(maximum > 0.9)),
        scope="Unfamiliar-class confidence only; no unknown/rejection rule",
    )


def discovery_metrics(y, labels, k, *, labeled):
    result = dict(selected_k=k, assigned_units=len(labels))
    if y is not None:
        result["ari"] = float(adjusted_rand_score(y, labels))
    if labeled:
        result["four_group_recovery"] = bool(k == 4 and result["ari"] >= 0.8)
    return result
