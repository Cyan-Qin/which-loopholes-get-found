"""Small statistics helpers shared by the audits and the analysis."""

import math


def wilson_interval(k, n, z=1.96):
    """Wilson score interval for a binomial proportion k/n (95% by default).

    Unlike the normal approximation it stays inside [0, 1] and behaves at k = 0,
    which matters for judges such as xVerify that almost never accept an attack.
    """
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))
