"""Unit tests for execution.batching.estimate_task_timeout_s."""
from __future__ import annotations

import pytest

from decoder_confidence.execution.batching import estimate_task_timeout_s


def test_estimate_task_timeout_s_uses_multiplier():
    # per_shot = 2.0s / 10 = 0.2s; expected_task = 0.2 * 50 = 10s; * 10 = 100s
    timeout = estimate_task_timeout_s(
        probe_shots=10,
        probe_duration_s=2.0,
        shots_per_task=50,
        timeout_multiplier=10.0,
        min_task_timeout_s=1.0,
    )
    assert timeout == pytest.approx(100.0)


def test_estimate_task_timeout_s_applies_floor():
    # per_shot = 1.0s / 100 = 0.01s; expected_task = 0.01 * 1 = 0.01s;
    # * 10 = 0.1s, well below the 60s floor.
    timeout = estimate_task_timeout_s(
        probe_shots=100,
        probe_duration_s=1.0,
        shots_per_task=1,
        timeout_multiplier=10.0,
        min_task_timeout_s=60.0,
    )
    assert timeout == 60.0


def test_estimate_task_timeout_s_rejects_nonpositive_probe():
    with pytest.raises(ValueError):
        estimate_task_timeout_s(
            probe_shots=0,
            probe_duration_s=1.0,
            shots_per_task=10,
            timeout_multiplier=10.0,
            min_task_timeout_s=60.0,
        )

    with pytest.raises(ValueError):
        estimate_task_timeout_s(
            probe_shots=10,
            probe_duration_s=0.0,
            shots_per_task=10,
            timeout_multiplier=10.0,
            min_task_timeout_s=60.0,
        )
