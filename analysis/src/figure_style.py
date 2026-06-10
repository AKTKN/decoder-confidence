"""Shared figure styling utilities.

This module separates *style* (colors, tick direction, font sizes, paper
formatting) from the data-plotting code in the rest of :mod:`analysis.src`.

Typical notebook usage::

    from analysis.src import apply_paper_style, finalize_axes, finalize_plot

    apply_paper_style()  # call once, in the first cell

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    NumericMetricAnalyzer().plot_distribution(lf, config, ax)
    ax.set_title("...")
    ax.set_xlabel("...")
    ax.set_ylabel("...")
    ax.legend()
    finalize_plot(fig, ax)  # applies finalize_axes() + show/save
"""
from __future__ import annotations

from typing import Callable, Iterable

import matplotlib.pyplot as plt
from cycler import cycler

#: tab10 colors as hex strings, in the standard matplotlib order.
TAB10_COLORS: list[str] = [
    "#%02x%02x%02x" % tuple(int(round(c * 255)) for c in rgb)
    for rgb in plt.get_cmap("tab10").colors
]


def set_tab10_colors() -> None:
    """Set the default axes color cycle to tab10.

    This is matplotlib's default color cycle already, but calling this
    explicitly guards against other style sheets overriding it.
    """
    plt.rcParams["axes.prop_cycle"] = cycler(color=TAB10_COLORS)


def apply_paper_style() -> None:
    """Apply paper-friendly rcParams (fonts, tick direction, tab10 colors).

    Intended to be called once, in the first cell of a notebook. Any of
    these settings can be overridden afterwards on a per-figure or per-axes
    basis — notebook settings take priority.
    """
    set_tab10_colors()
    plt.rcParams.update({
        "axes.labelsize": "large",
        "axes.titlesize": "large",
        "xtick.labelsize": "large",
        "ytick.labelsize": "large",
        "legend.fontsize": "large",
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "axes.linewidth": 1.0,
        "text.usetex": True,
        "font.family": "serif",
    })


def finalize_axes(ax: plt.Axes) -> None:
    """Apply common per-axes touch-ups right before a figure is shown/saved.

    - Ticks point inward on all four sides.
    - Spines have a consistent line width.
    - Spines are pushed *behind* plotted data (zorder), so that data points
      lying on the axes frame are drawn on top of it instead of being
      partially hidden.
    """
    ax.tick_params(direction="in", length=4, width=1.0, top=True, right=True)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_zorder(0.5)


class MultiStyleFigureExporter:
    """Render a plotting callback into multiple paper-formatted PDF styles.

    Optional utility for exporting the same figure as both an "arxiv" and a
    "thesis" formatted PDF (via :mod:`rsmf`). For everyday notebook use,
    prefer :func:`apply_paper_style` + :func:`finalize_axes` /
    :func:`~analysis.src.plot_utils.finalize_plot`.
    """

    def __init__(self) -> None:
        import rsmf

        self.formatters = {
            "arxiv": rsmf.setup(r"\documentclass[a4paper,reprint,unpublished]{revtex4-2}"),
            "thesis": rsmf.setup(r"\documentclass[a4paper,preprint,unpublished]{revtex4-2}"),
        }

    def export_all(
        self,
        plot_func: Callable[[plt.Axes], None],
        base_filename: str,
        aspect_ratio: float = 0.75,
    ) -> None:
        """Generate and save one PDF per registered style.

        Parameters
        ----------
        plot_func:
            Callback that draws onto the given axes (data only — set
            title/labels/legend inside the callback if needed for this
            export).
        base_filename:
            Output filename base (without extension).
        aspect_ratio:
            Figure aspect ratio (height / width).
        """
        for style_name, formatter in self.formatters.items():
            fig = formatter.figure(aspect_ratio=aspect_ratio)
            ax = fig.add_subplot(111)

            plot_func(ax)
            finalize_axes(ax)

            output_path = f"{base_filename}_{style_name}.pdf"
            fig.savefig(output_path, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved: {output_path}")
