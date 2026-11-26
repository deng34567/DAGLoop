"""Tests for the shared executable I/O format contract (check_shape). Pure, stdlib."""
from __future__ import annotations

from autoresearch.experiments.general.contract import check_shape, shape_signature


def test_matching_shapes_pass():
    # list of bins, each a list of int indices
    assert check_shape([[0, 1], [2]], [[0], [1, 2]])[0] is True
    assert check_shape([0, 1, 2], [3, 4])[0] is True
    assert check_shape({"bins": [[0]]}, {"bins": [[1, 2]]})[0] is True


def test_indices_vs_values_caught():
    # THE bug: example is int indices; a method returning floats (values) is caught.
    ok, reason = check_shape([[0.5, 1.2], [0.7]], [[0, 1], [2]])
    assert ok is False
    assert "expected int, got float" in reason


def test_int_accepted_where_float_expected():
    # A whole-number int is fine where the example is a float value.
    assert check_shape([1, 2], [1.0, 2.0])[0] is True


def test_wrong_nesting_caught():
    # flat list where the example is a list of lists (bins)
    ok, reason = check_shape([0, 1, 2], [[0], [1, 2]])
    assert ok is False
    assert "expected list, got int" in reason


def test_dict_vs_list_caught():
    ok, reason = check_shape([0, 1], {"assignment": [0, 1]})
    assert ok is False
    assert "expected dict, got list" in reason


def test_missing_dict_key_caught():
    ok, reason = check_shape({"a": 1}, {"a": 1, "bins": [[0]]})
    assert ok is False
    assert "missing key" in reason and "bins" in reason


def test_empty_list_is_wildcard():
    assert check_shape([], [[0], [1]])[0] is True
    assert check_shape([[0]], [])[0] is True


def test_none_example_accepts_anything():
    assert check_shape([[0]], None)[0] is True


def test_none_solution_caught():
    ok, reason = check_shape(None, [[0]])
    assert ok is False
    assert "expected list, got none" in reason


def test_shape_signature_readable():
    assert shape_signature([[0, 1]]) == "list[list[int]]"
    assert shape_signature([0.5]) == "list[float]"


# --- anti-cheat b: strip instances to the declared schema ----------------------

def test_strip_to_contract_drops_extra_fields():
    from autoresearch.experiments.general.contract import strip_to_contract
    assert strip_to_contract({"n": 1, "_secret": 999}, {"n": 0}) == {"n": 1}
    assert strip_to_contract({"n": 1}, {"n": 0}) == {"n": 1}          # nothing extra
    assert strip_to_contract([1, 2], {"n": 0}) == [1, 2]              # non-dict unchanged
    assert strip_to_contract({"n": 1}, None) == {"n": 1}             # no example -> unchanged
