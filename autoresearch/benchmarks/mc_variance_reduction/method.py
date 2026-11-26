"""The ONLY editable file. Baseline = naive Monte Carlo via ``sample_f``.

The optimization goal is to LOWER the RMSE of the estimate using the SAME (or
fewer) budgeted ``sample_f`` calls -- e.g. importance sampling, control
variates, or antithetic variates. ``sample_f`` is the only window onto the
integrand and it counts calls: making more than ``n_samples`` calls scores 0.
"""


def proposed_estimate(problem, sample_f, n_samples, rng):
    d = problem["dim"]
    return sum(
        sample_f([rng.random() for _ in range(d)]) for _ in range(n_samples)
    ) / n_samples
