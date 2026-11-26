"""FIXED authoring-time shape self-test (D11) -- run in the METHOD-writer's
container right after the agent writes method.py, to ENFORCE the shape contract
even if the agent skipped its self-test.

Reads a concrete example from ``_shape_example.json`` ({"instance": ..., "solution":
...}), calls the AI-written ``method.solve(instance)`` and checks its shape against
the example ``solution`` via the trusted, leak-safe ``contract.check_shape``. Prints
one line: ``SHAPE_OK`` or ``SHAPE_FAIL: <leak-safe reason>``. Stdlib + the shipped
contract.py only; it reveals NOTHING about the ruler (only the public shape example).
"""
import json
import sys


def main() -> None:
    try:
        ex = json.load(open("_shape_example.json"))
        inst, sol_ex = ex.get("instance"), ex.get("solution")
    except Exception as exc:  # noqa: BLE001 -- no example -> nothing to check
        print("SHAPE_SKIP: " + str(exc))
        return
    try:
        import contract
        import method
        sol = method.solve(inst)
        ok, why = contract.check_shape(sol, sol_ex)
    except Exception as exc:  # noqa: BLE001 -- the method's OWN error (leak-safe)
        print(f"SHAPE_FAIL: solve() raised {type(exc).__name__}: {exc}")
        return
    print("SHAPE_OK" if ok else "SHAPE_FAIL: " + (why or "wrong shape"))


if __name__ == "__main__":
    main()
    sys.stdout.flush()
