"""Problem metadata handed to ``method.py`` (dimension / sample budget / seeds).

This deliberately carries NO integration function -- the method only sees these
knobs and reaches the integrand through the evaluator's budgeted ``sample_f``.
"""

DEFAULT_PROBLEM = {
    "dim": 4,
    "n_samples": 4000,
    "seeds": [0, 1, 2, 3, 4, 5, 6, 7],
    "domain": "unit_cube",
    "name": "oscillatory_ratio_4d",
}
