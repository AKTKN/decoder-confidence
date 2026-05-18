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
