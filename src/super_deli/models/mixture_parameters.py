"""Dependency-free dimensions for the supported integrative Gaussian families."""


def mixture_parameter_count(k, d, covariance_type="full"):
    if type(k) is not int or type(d) is not int or k < 1 or d < 1:
        raise ValueError("Positive integer component and score dimensions are required")
    covariance = d * (d + 1) // 2
    if covariance_type == "full":
        return k * (d + covariance) + k - 1
    if covariance_type == "tied":
        return k * d + covariance + k - 1
    raise ValueError("Supported covariance types are full and tied")
