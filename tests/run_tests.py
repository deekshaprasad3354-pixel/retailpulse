"""Run the RetailPulse test-suite without pytest.

The suite is written with pytest-style classes and assertions, but pytest itself
is an optional dependency here. This runner discovers every ``Test*`` class in
``tests/``, executes each ``test_*`` method, and reports pass/fail counts.

    python tests/run_tests.py

When pytest IS available, prefer ``pytest tests -v`` — it gives better output.
"""
from __future__ import annotations

import inspect
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402


class _Approx:
    """Minimal stand-in for pytest.approx.

    Implements both comparison directions: ``value == approx(x)`` first calls
    ``value.__eq__``, which returns ``NotImplemented`` for an unknown type, and
    Python then falls back to the reflected ``approx.__eq__(value)``.
    """

    def __init__(self, expected, rel: float = 1e-6, abs: float = 1e-12):
        self.expected = expected
        self.rel = rel
        self.abs = abs

    def __eq__(self, other):
        try:
            return bool(np.isclose(other, self.expected, rtol=self.rel, atol=self.abs,
                                   equal_nan=False))
        except (TypeError, ValueError):
            return NotImplemented

    def __hash__(self):
        return hash(self.expected)

    def __repr__(self):
        return f"approx({self.expected})"


class _PytestShim:
    """Just enough of the pytest API for this suite to run standalone."""

    approx = staticmethod(lambda value, rel=1e-6, abs=1e-12: _Approx(value, rel, abs))

    class mark:  # noqa: N801
        @staticmethod
        def parametrize(argnames, argvalues):
            def decorator(func):
                func._parametrize = (argnames, argvalues)
                return func
            return decorator


def _install_shim() -> None:
    if "pytest" not in sys.modules:
        try:
            import pytest  # noqa: F401
        except ImportError:
            sys.modules["pytest"] = _PytestShim()  # type: ignore[assignment]


def _expand_params(func) -> list[dict]:
    spec = getattr(func, "_parametrize", None)
    if not spec:
        return [{}]
    names, values = spec
    names = [n.strip() for n in names.split(",")]
    cases = []
    for value in values:
        values_tuple = value if isinstance(value, (tuple, list)) else (value,)
        cases.append(dict(zip(names, values_tuple)))
    return cases


def main() -> int:
    _install_shim()
    import importlib

    module = importlib.import_module("tests.test_retailpulse")

    passed = failed = skipped = 0
    failures: list[tuple[str, str]] = []

    classes = [obj for name, obj in vars(module).items()
               if name.startswith("Test") and inspect.isclass(obj)]

    for cls in classes:
        instance = cls()
        methods = [m for m in dir(cls) if m.startswith("test_")]
        for method_name in sorted(methods):
            func = getattr(instance, method_name)
            for params in _expand_params(getattr(cls, method_name)):
                label = f"{cls.__name__}.{method_name}"
                if params:
                    label += f"[{', '.join(f'{k}={v!r}' for k, v in params.items())}]"
                try:
                    func(**params)
                    passed += 1
                    print(f"  PASS  {label}")
                except Exception as exc:  # noqa: BLE001
                    failed += 1
                    failures.append((label, traceback.format_exc()))
                    print(f"  FAIL  {label}: {type(exc).__name__}: {exc}")

    print("\n" + "=" * 74)
    print(f"RESULT: {passed} passed, {failed} failed, {skipped} skipped")
    print("=" * 74)

    if failures:
        print("\nFailures:")
        for label, tb in failures:
            print(f"\n--- {label} ---")
            print(tb)

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
