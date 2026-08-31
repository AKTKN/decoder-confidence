"""Shared pytest fixtures for the decoder-confidence test suite.

Gurobi opt-in gate
------------------
A handful of tests build a *real* ``gurobipy.Env``/``ILPDecoder`` (as opposed
to a mocked one) to check ILP-decoder behavior end-to-end. Those Gurobi API
calls are slow and are metered against a per-account limit shared with other
work, so they must never run unless explicitly requested. Tests that need
this must depend on the ``_gurobi_opt_in`` fixture below (typically via a
decoder-specific fixture such as ``gurobi_env`` depending on it), which fails
loudly -- rather than silently skipping -- when the opt-in is absent, so an
accidental full-suite run cannot burn API quota unnoticed.

Opt in with: ``RUN_GUROBI_TESTS=1 pytest -m gurobi`` (accepted values are
"1"/"true"/"t"/"yes"/"y"/"run", case-insensitive).
"""
from __future__ import annotations

import os

import pytest

_GUROBI_OPT_IN_VAR = "RUN_GUROBI_TESTS"
_TRUTHY = {"1", "true", "t", "yes", "y", "run"}


def gurobi_tests_enabled() -> bool:
    """Whether tests that call the real Gurobi API are allowed to run."""
    return os.environ.get(_GUROBI_OPT_IN_VAR, "").strip().lower() in _TRUTHY


@pytest.fixture(scope="session")
def _gurobi_opt_in() -> None:
    """Fail (not skip) any dependent test unless RUN_GUROBI_TESTS is set.

    Depend on this fixture -- directly, or transitively via a fixture like
    ``gurobi_env`` -- in any test that constructs a real Gurobi environment
    or decoder.
    """
    if not gurobi_tests_enabled():
        pytest.fail(
            "This test calls the real Gurobi API and is disabled by default "
            "(Gurobi API calls are slow and rate/seat-limited on the shared "
            f"account). Set {_GUROBI_OPT_IN_VAR}=1 to run it intentionally, "
            "e.g.: RUN_GUROBI_TESTS=1 pytest -m gurobi"
        )
