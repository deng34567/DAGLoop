"""TRUSTED fixed baseline (NOT editable): naive Monte Carlo via ``sample_f``.

``run.py`` scores the editable ``method.py`` against THIS file, so the reference
baseline is always trusted code -- never whatever the agent left in method.py.
``method.py`` starts life as a copy of this; the agent optimizes that copy.
"""


def proposed_estimate(problem, sample_f, n_samples, rng):
    d = problem["dim"]
    return sum(
        sample_f([rng.random() for _ in range(d)]) for _ in range(n_samples)
    ) / n_samples
