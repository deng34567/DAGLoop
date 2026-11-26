"""Toy benchmark task (the ONLY file in the editable_region).

Compute S = sum over all pairs i<j of data[i]*data[j].

The baseline below is the naive O(N^2) double loop. The optimization goal is to
make ``solve`` faster (e.g. via the identity S = (sum(data)^2 - sum(x^2)) / 2,
which is O(N)) WITHOUT changing the result. run.py checks correctness against a
trusted reference it computes itself, so a wrong-but-fast solution scores 0.
"""


def solve(data):
    total = 0
    n = len(data)
    for i in range(n):
        di = data[i]
        for j in range(i + 1, n):
            total += di * data[j]
    return total
