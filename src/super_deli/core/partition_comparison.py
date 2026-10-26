"""Label-invariant, coverage-explicit agreement; noise is never a shared cluster."""

from collections import Counter
from fractions import Fraction


def _pairs(n):
    return n * (n - 1) // 2


def compare_partitions(left, right, left_resolved, right_resolved):
    """Compare aligned partitions conditional on both units having resolved labels.

    Integer labels are arbitrary. -1 denotes noise. A degenerate ARI of one is
    the standard mathematical convention, not evidence of recovered structure.
    No unit-pair matrix is allocated.
    """
    n = len(left)
    if not all(len(x) == n for x in (right, left_resolved, right_resolved)):
        raise ValueError("Aligned partitions and masks must have equal lengths")
    for labels in (left, right):
        if any(type(v) is not int or v < -1 for v in labels):
            raise ValueError("Partition labels must be integers >= -1")
    for mask in (left_resolved, right_resolved):
        if any(type(v) is not bool for v in mask):
            raise ValueError("Resolved masks must be boolean")
    li = [ok and v >= 0 for ok, v in zip(left_resolved, left, strict=True)]
    ri = [ok and v >= 0 for ok, v in zip(right_resolved, right, strict=True)]
    joint = [i for i in range(n) if li[i] and ri[i]]
    a, b = [left[i] for i in joint], [right[i] for i in joint]
    ca, cb, cells = Counter(a), Counter(b), Counter(zip(a, b, strict=True))
    total = _pairs(len(joint))
    same_a, same_b, same_both = (sum(_pairs(v) for v in c.values()) for c in (ca, cb, cells))
    discordant = same_a + same_b - 2 * same_both
    ari = None
    if total:
        expected = Fraction(same_a * same_b, total)
        denominator = Fraction(same_a + same_b, 2) - expected
        ari = float((same_both - expected) / denominator) if denominator else 1.0
    return {
        "status": "available" if total else "undefined",
        "n_units": n,
        "left_resolved": sum(li),
        "right_resolved": sum(ri),
        "joint_resolved": len(joint),
        "joint_coverage": len(joint) / n if n else None,
        "joint_pairs": total,
        "discordant_pairs": discordant,
        "coassignment_agreement": 1 - discordant / total if total else None,
        "adjusted_rand": ari,
        "left_clusters_on_intersection": len(ca),
        "right_clusters_on_intersection": len(cb),
        "degenerate": len(ca) < 2 or len(cb) < 2 or not total,
        "interpretation": "Conditional on jointly resolved non-noise units; no accuracy claim",
    }
