"""
Differentially private percentiles, minimum and maximum.

Method: the exponential mechanism over the intervals between sorted data
points (Smith, 2011; the same approach used in Google's DP library).

- Data are clipped to [lower, upper]. These bounds must be chosen WITHOUT
  looking at the data (or be public knowledge); otherwise they leak privacy.
- Each function satisfies epsilon-DP (add/remove or replace-one neighbours)
  for the epsilon you pass in.
"""

import numpy as np


def dp_quantile(data, q, epsilon, lower, upper, rng=None):
    """Return an epsilon-DP estimate of the q-th quantile (0 <= q <= 1)."""
    if not 0 <= q <= 1:
        raise ValueError("q must be in [0, 1]")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    if lower >= upper:
        raise ValueError("lower must be less than upper")

    rng = np.random.default_rng(rng)
    x = np.sort(np.clip(np.asarray(data, dtype=float), lower, upper))
    n = len(x)

    # Interval i is [edges[i], edges[i+1]]; there are n + 1 of them.
    edges = np.concatenate(([lower], x, [upper]))
    widths = np.diff(edges)

    # Utility of interval i = -|i - q*n| (distance from the target rank).
    # Changing one record shifts any rank by at most 1, so sensitivity = 1.
    utility = -np.abs(np.arange(n + 1) - q * n)

    # log-weight = log(width) + epsilon * utility / (2 * sensitivity)
    with np.errstate(divide="ignore"):
        log_w = np.log(widths) + epsilon * utility / 2.0  # log(0) = -inf is fine
    log_w -= log_w.max()  # numerical stability
    probs = np.exp(log_w)
    probs /= probs.sum()

    i = rng.choice(n + 1, p=probs)
    return float(rng.uniform(edges[i], edges[i + 1]))


def dp_percentile(data, percentile, epsilon, lower, upper, rng=None):
    """Epsilon-DP percentile, with percentile in [0, 100]."""
    return dp_quantile(data, percentile / 100.0, epsilon, lower, upper, rng)


def dp_percentiles(data, percentiles, epsilon, lower, upper, rng=None):
    """
    Several percentiles at once. The total budget `epsilon` is split evenly
    (basic composition), so each percentile gets epsilon / len(percentiles).
    Returns a dict {percentile: estimate}.
    """
    rng = np.random.default_rng(rng)
    eps_each = epsilon / len(percentiles)
    return {
        p: dp_percentile(data, p, eps_each, lower, upper, rng) for p in percentiles
    }


def dp_min(data, epsilon, lower, upper, rng=None):
    """Epsilon-DP minimum (the 0th quantile)."""
    return dp_quantile(data, 0.0, epsilon, lower, upper, rng)


def dp_max(data, epsilon, lower, upper, rng=None):
    """Epsilon-DP maximum (the 100th quantile)."""
    return dp_quantile(data, 1.0, epsilon, lower, upper, rng)


if __name__ == "__main__":
    #All of this was generated raw by Claude, keeping it felt appropriate
    #There was a major time gap between prompts for LLM_Implementation.py and LLM_ImplementationTwo.py, and as such
    #Claude forgot the project context, and the functions are what is most important. They are unchanged but the calling differs.
    rng = np.random.default_rng(42)
    ages = rng.normal(40, 12, size=5000).clip(18, 90)
    #The true values are already known from the Traditional Implementation file. They will not be recalculated.
    """
    print("True median :", round(float(np.median(ages)), 2))
    print("DP median   :", round(dp_percentile(ages, 50, 1.0, 0, 100, rng), 2))
    print("True min/max:", round(float(ages.min()), 2), round(float(ages.max()), 2))
    print("DP min      :", round(dp_min(ages, 1.0, 0, 100, rng), 2))
    print("DP max      :", round(dp_max(ages, 1.0, 0, 100, rng), 2))
    print("DP quartiles (total eps=1):", dp_percentiles(ages, [25, 50, 75], 1.0, 0, 100, rng))
    """
