"""TRUSTED integrand for the mc_variance_reduction benchmark.

Only ``run.py`` / the evaluator import this. The optimizable ``method.py`` must
NEVER import it directly -- it may only touch ``f`` through the budgeted
``sample_f`` the evaluator hands it (which counts calls). See the plan's threat
model: budgeted access + no closed form is what makes importance sampling /
control variates the honest way to win.

``f`` is smooth (so the evaluator's quadrature can compute a trusted truth) and
NON-separable with no elementary closed form (so naive Monte Carlo has real
variance to reduce, and "just integrate it densely yourself" is blocked by the
sample budget).
"""
import math


def integrand(x):
    """x: list[float] in [0,1]^d -> float."""
    s = sum(x)
    q = sum(xi * xi for xi in x)
    return math.exp(math.sin(2.0 * math.pi * s)) / (1.0 + q)
