"""Example: post-selection performance curves for multiple metrics.

Run from the repository root::

    python -m analysis.examples.plot_postselect

Plots abort rate vs post-selected logical error rate for three metrics on a
single Axes, with Wilson 95 % CI shading.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from analysis.src import (
    PostSelectionPlotter,
    PostSelectSpec,
    SimulationDataManager,
)


RESULT_DIR = Path(__file__).parents[2] / "raw_data"


def main() -> None:
    manager = SimulationDataManager(RESULT_DIR)

    base_filters = {"d": 7, "p": 0.1}

    specs = [
        PostSelectSpec(
            metric_name="logical_gap",
            decoder_names=["ILP"],
            filters=base_filters,
        ),
        PostSelectSpec(
            metric_name="linearize_logicalgap",
            decoder_names=["MWPM"],
            filters=base_filters,
        ),
        PostSelectSpec(
            metric_name="cluster_llr",
            decoder_names=["BP-LSD"],
            filters=base_filters,
        ),
    ]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_title(f"Post-selection performance (d={base_filters['d']}, p={base_filters['p']})")
    ax.set_yscale("log")

    PostSelectionPlotter(manager).plot(specs, ax, num_points=50, alpha=0.05)

    fig.tight_layout()
    out = Path(__file__).parent / "postselect_d7.png"
    fig.savefig(out, dpi=200)
    print(f"Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
