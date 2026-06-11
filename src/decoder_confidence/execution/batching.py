from __future__ import annotations


def estimate_shots_per_task(
    probe_shots: int,
    probe_duration_s: float,
    target_duration_s: float,
    *,
    min_shots: int = 1,
    max_shots: int | None = None,
) -> int:
    if probe_shots <= 0:
        raise ValueError(f"probe_shots must be > 0 but got {probe_shots}")
    if probe_duration_s <= 0:
        raise ValueError(
            f"probe_duration_s must be > 0 but got {probe_duration_s}"
        )
    if target_duration_s <= 0:
        raise ValueError(
            f"target_duration_s must be > 0 but got {target_duration_s}"
        )

    shots = int((probe_shots / probe_duration_s) * target_duration_s)
    if shots < min_shots:
        shots = min_shots
    if max_shots is not None and shots > max_shots:
        shots = max_shots
    return shots


def estimate_task_timeout_s(
    probe_shots: int,
    probe_duration_s: float,
    shots_per_task: int,
    *,
    timeout_multiplier: float,
    min_task_timeout_s: float,
) -> float:
    """Derive an adaptive per-task timeout from the probe measurement.

    expected_task_s = (probe_duration_s / probe_shots) * shots_per_task
    timeout = max(timeout_multiplier * expected_task_s, min_task_timeout_s)
    """
    if probe_shots <= 0:
        raise ValueError(f"probe_shots must be > 0 but got {probe_shots}")
    if probe_duration_s <= 0:
        raise ValueError(
            f"probe_duration_s must be > 0 but got {probe_duration_s}"
        )

    per_shot_s = probe_duration_s / probe_shots
    expected_task_s = per_shot_s * shots_per_task
    return max(timeout_multiplier * expected_task_s, min_task_timeout_s)
