"""
dp_stats.py - Laplace-mechanism differential privacy for mean, median and
standard deviation of a numeric column.
 
Assumptions (read these before using the results):
  * Bounded DP: neighboring datasets differ by REPLACING one record, and the
    dataset size n is treated as public.
  * You supply public bounds (lower, upper). Values are clipped to them, so
    choose bounds without looking at the private data if you can.
  * Each function spends the epsilon you give it. Calling several functions on
    the same data spends the SUM of their epsilons (sequential composition).
"""
import numpy as np

 
 
def _prep(data, lower, upper):
    if lower >= upper:
        raise ValueError("lower must be less than upper")
    x = np.clip(np.asarray(data, dtype=float), lower, upper)
    if x.size == 0:
        raise ValueError("data is empty")
    return x
 
 
def dp_mean(data, epsilon, lower, upper, rng=None):
    """Laplace-noised mean. Sensitivity = (upper - lower) / n."""
    rng = rng or np.random.default_rng()
    x = _prep(data, lower, upper)
    sensitivity = (upper - lower) / len(x)
    return float(x.mean() + rng.laplace(0, sensitivity / epsilon))
 
 
def dp_median(data, epsilon, lower, upper, bin_width=1.0, rng=None):
    """
    Median from a Laplace-noised histogram.
 
    Adding noise straight to the median is useless (its sensitivity is the
    whole range), so we noise the bin counts instead (L1 sensitivity 2 under
    replace-one) and read the median off the noisy cumulative counts,
    interpolating within the bin. Smaller bins = finer resolution but more
    accumulated noise.
    """
    rng = rng or np.random.default_rng()
    x = _prep(data, lower, upper)
    n = len(x)
    edges = np.arange(lower, upper + bin_width, bin_width)
    if edges[-1] < upper:
        edges = np.append(edges, upper)
    counts, _ = np.histogram(x, bins=edges)
    noisy = np.maximum(counts + rng.laplace(0, 2.0 / epsilon, len(counts)), 0)
    cum = np.cumsum(noisy)
    k = min(int(np.searchsorted(cum, n / 2)), len(noisy) - 1)
    prev = cum[k - 1] if k > 0 else 0.0
    frac = (n / 2 - prev) / noisy[k] if noisy[k] > 0 else 0.5
    frac = float(np.clip(frac, 0, 1))
    return float(edges[k] + frac * (edges[k + 1] - edges[k]))
 
 
def dp_std(data, epsilon, lower, upper, dp_mean_value=None, rng=None):
    """
    Std dev via Var = E[x^2] - mean^2.
 
    Spends `epsilon` on a Laplace-noised mean of squares (sensitivity
    (upper^2 - lower^2) / n when lower >= 0). You must pass an already-released
    DP mean (post-processing, no extra cost); if omitted, one is released
    using the same epsilon, so the total spend is 2 * epsilon.
    Negative variance estimates are clamped to 0.
    """
    rng = rng or np.random.default_rng()
    if lower < 0:
        # shift so the squared range is well-behaved
        data = np.asarray(data, dtype=float) - lower
        if dp_mean_value is not None:
            dp_mean_value -= lower
        upper, lower = upper - lower, 0.0
    x = _prep(data, lower, upper)
    n = len(x)
    if dp_mean_value is None:
        dp_mean_value = dp_mean(x, epsilon, lower, upper, rng)
    sensitivity = (upper ** 2 - lower ** 2) / n
    m2 = float((x ** 2).mean() + rng.laplace(0, sensitivity / epsilon))
    return float(np.sqrt(max(m2 - dp_mean_value ** 2, 0.0)))
 
 
def dp_summary(data, epsilon, lower, upper, split=(1, 1, 1), bin_width=1.0, seed=None):
    """
    Release mean, median and std together, spending `epsilon` in total.
    `split` gives the relative budget for (mean, median, std).
    """
    rng = np.random.default_rng(seed)
    w = np.asarray(split, dtype=float)
    e_mean, e_med, e_std = epsilon * w / w.sum()
    print("Summary Epsilon is: ", str(e_mean))
    m = dp_mean(data, e_mean, lower, upper, rng)
    med = dp_median(data, e_med, lower, upper, bin_width, rng)
    sd = dp_std(data, e_std, lower, upper, dp_mean_value=m, rng=rng)
    return {"mean": m, "median": med, "std": sd, "epsilon_spent": float(e_mean + e_med + e_std)}



"""
Differentially private percentiles, minimum and maximum.

Method: the exponential mechanism over the intervals between sorted data
points (Smith, 2011; the same approach used in Google's DP library).

- Data are clipped to [lower, upper]. These bounds must be chosen WITHOUT
  looking at the data (or be public knowledge); otherwise they leak privacy.
- Each function satisfies epsilon-DP (add/remove or replace-one neighbours)
  for the epsilon you pass in.
"""


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
 
#This part is not AI-generated, and is just how the code itself is run.

if __name__ == "__main__":
    import pandas as pd
    import matplotlib.pyplot as plt
 
    fares = pd.read_csv("Datasets/titanic_clean.csv")["Fare"]

    epsilon = 0.8
    lower   = 0
    upper   = 600
    seed    = 42
    rng     = np.random.default_rng(seed)

    #print(dp_summary(fares, epsilon=0.8, lower=0, upper=513, seed=42))

    #Here I find the various stats for the data.
    privatemean         = dp_mean(fares, epsilon, lower, upper, rng)
    privatemedian       = dp_median(fares, epsilon, lower, upper, 1.0 , rng)
    privatestd          = dp_std(fares, epsilon, lower, upper, dp_mean_value=privatemean, rng=rng)
    privatemin          = dp_min(fares, epsilon, lower, upper, rng)
    privatemax          = dp_max(fares, epsilon, lower, upper, rng)
    privatetwentyfifth  = dp_percentile(fares, 25, epsilon, lower, upper, rng)
    privateseventyfifth = dp_percentile(fares, 75, epsilon, lower, upper, rng)



    #Here I run each several times, just to gauge accuracy
    privatemeantotal         = float(0)
    privatemediantotal       = float(0)
    privatestdtotal          = float(0)
    privatemintotal          = float(0)
    privatemaxtotal          = float(0)
    privatetwentyfifthtotal  = float(0)
    privateseventyfifthtotal = float(0)
    
    totalruns = 100    

    for i in range(1, 100):
        privatemeantotal         = privatemeantotal         + dp_mean(fares, epsilon, lower, upper, rng)
        privatemediantotal       = privatemediantotal       + dp_median(fares, epsilon, lower, upper, 1.0 , rng)
        privatestdtotal          = privatestdtotal          + dp_std(fares, epsilon, lower, upper, rng=rng)
        privatemintotal          = privatemintotal          + dp_min(fares, epsilon, lower, upper, rng)
        privatemaxtotal          = privatemaxtotal          + dp_max(fares, epsilon, lower, upper, rng)
        privatetwentyfifthtotal  = privatetwentyfifthtotal  + dp_percentile(fares, 25, epsilon, lower, upper, rng)
        privateseventyfifthtotal = privateseventyfifthtotal + dp_percentile(fares, 75, epsilon, lower, upper, rng)

    avgmean         = float(privatemeantotal         / totalruns)
    avgmedian       = float(privatemediantotal       / totalruns)
    avgstd          = float(privatestdtotal          / totalruns)
    avgmin          = float(privatemintotal          / totalruns)
    avgmax          = float(privatemaxtotal          / totalruns)
    avgtwentyfifth  = float(privatetwentyfifthtotal  / totalruns)
    avgseventyfifth = float(privateseventyfifthtotal / totalruns)

    with open("Results/LLM_Implementation.csv", "w") as f:
        f.write(f"Statistic,LLM_Private,Average_LLM_Private\n")
        f.write(f"Mean,{privatemean},{avgmean}\n")
        f.write(f"Median,{privatemedian},{avgmedian}\n")
        f.write(f"STD,{privatestd},{avgstd}\n")
        f.write(f"25th Percentile,{privatetwentyfifth},{avgtwentyfifth}\n")
        f.write(f"75th Percentile,{privateseventyfifth},{avgseventyfifth}\n")
        f.write(f"Minimum,{privatemin},{avgmin}\n")
        f.write(f"Maximum,{privatemax},{avgmax}\n")

    