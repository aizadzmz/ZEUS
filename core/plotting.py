from typing import List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from core.io_utils import EISDataset

# The gold crosshair marking Z'=0 and -Z''=0 on Nyquist plots, drawn heavier
# than the grid so the origin reads as a reference the eye can find rather
# than as another gridline. core.plotting_pg imports these, so both the
# matplotlib and PyQtGraph backends stay in sync from one place.
ORIGIN_COLOR = "#CC9D33"
ORIGIN_WIDTH = 2.0


def plot_single(
    dataset: EISDataset,
    title: Optional[str] = None,
    show: bool = True,
    save_path: Optional[str] = None,
    style: str = "scatter",
    show_removed: bool = False,
) -> tuple[Figure, Axes]:
    """
    Plot a Nyquist plot for a single EISDataset.

    Args:
        dataset      : A single EISDataset to plot.
        title        : Plot title. Defaults to the dataset's full label.
        show         : Whether to call plt.show().
        save_path    : If provided, saves the figure to this path.
        style        : "scatter" or "line".
        show_removed : If True, also plot masked-out points as muted markers.

    Returns:
        fig, ax : The Figure and Axes objects.
    """

    fig, ax = plt.subplots(figsize=(8, 6))

    Z = dataset.impedances

    _plot_series(ax, Z, dataset.frequencies, dataset.label, style)

    if show_removed:
        _plot_removed(
            ax,
            dataset.data.get_impedances(masked=True),
            dataset.data.get_frequencies(masked=True),
            dataset.label,
            "Removed",
        )

    ax.set_title(title or dataset.full_label, fontsize=12)
    fig.canvas.manager.set_window_title(dataset.full_label)
    _format_ax(ax)

    plt.tight_layout(rect=[0, 0, 0.85, 1])

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    if show:
        plt.show()

    return fig, ax


def plot_overlay(
    datasets: List[EISDataset],
    title: str = "Nyquist Plot",
    show: bool = True,
    save_path: Optional[str] = None,
    style: str = "scatter",
    show_removed: bool = False,
) -> tuple[Figure, Axes]:
    """
    Overlay multiple EISDatasets on a single Nyquist plot.
    Each dataset gets its own color and legend entry.

    Args:
        datasets     : List of EISDataset objects to overlay.
        title        : Plot title.
        show         : Whether to call plt.show().
        save_path    : If provided, saves the figure to this path.
        style        : "scatter" or "line".
        show_removed : If True, also plot masked-out points as muted markers
                       (one shared "Removed" legend entry for all datasets).

    Returns:
        fig, ax : The Figure and Axes objects.
    """

    if not datasets:
        raise ValueError("No datasets provided to plot.")

    fig, ax = plt.subplots(figsize=(8, 7))

    for ds in datasets:
        Z = ds.impedances
        _plot_series(ax, Z, ds.frequencies, ds.label, style)

    if show_removed:
        removed_label = "Removed"
        for ds in datasets:
            _plot_removed(
                ax,
                ds.data.get_impedances(masked=True),
                ds.data.get_frequencies(masked=True),
                ds.label,
                removed_label,
            )
            removed_label = None  # only the first gets a legend entry

    window_title = datasets[0].source_file
    ax.set_title(title or window_title, fontsize=12)
    fig.canvas.manager.set_window_title(window_title)
    _format_ax(ax)

    plt.tight_layout(rect=[0, 0, 0.85, 1])

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    if show:
        plt.show()

    return fig, ax


def plot_residuals(
    result,
    title: Optional[str] = None,
    threshold: Optional[float] = None,
    show: bool = True,
    save_path: Optional[str] = None,
) -> tuple[Figure, Axes]:
    """
    Plot the relative residuals (ΔZ'/|Z| and ΔZ''/|Z|, in percent) of a
    validation result (Kramers-Kronig or Z-HIT) against frequency.

    Args:
        result    : A KramersKronigResult or ZHITResult (anything with
                    get_residuals_data()).
        title     : Plot title.
        threshold : If provided, draw dashed lines at ±threshold percent.
        show      : Whether to call plt.show().
        save_path : If provided, saves the figure to this path.

    Returns:
        fig, ax : The Figure and the central (residuals vs frequency) Axes.
    """
    freq, res_re, res_im = result.get_residuals_data()

    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")

    re_color, im_color = "C0", "C1"

    ax.axhline(0, color="0.6", linewidth=0.8, zorder=1)
    if threshold is not None:
        ax.axhline(threshold, color="red", linestyle="--", linewidth=0.8,
                   label=f"±{threshold}% threshold")
        ax.axhline(-threshold, color="red", linestyle="--", linewidth=0.8)
    ax.semilogx(freq, res_re, "o-", ms=4, color=re_color, label="ΔZ' / |Z|")
    ax.semilogx(freq, res_im, "s-", ms=4, color=im_color, label="ΔZ'' / |Z|")
    ax.set_xlabel("Frequency (Hz)", fontsize=10)
    ax.set_ylabel("Relative residual (%)", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)
    if title:
        ax.set_title(title, fontsize=12)
    ax.legend(fontsize=8, loc="best")

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    if show:
        plt.show()

    return fig, ax


def plot_drt(
    results: List[Tuple[str, object]],
    title: str = "DRT",
    show: bool = True,
    save_path: Optional[str] = None,
) -> tuple[Figure, Axes]:
    """
    Plot gamma vs tau (distribution of relaxation times) for one or more
    DRT results.

    Args:
        results   : List of (label, result) pairs, where each result exposes
                    get_drt_data() -> (tau, gamma), e.g. core.drt.run_drt's
                    return value. BHTResult (core.drt.run_drt_bht) instead
                    returns (tau, gamma_re, gamma_im), estimated separately
                    from the real and imaginary parts; both are plotted.
        title     : Plot title.
        show      : Whether to call plt.show().
        save_path : If provided, saves the figure to this path.

    Returns:
        fig, ax : The Figure and Axes objects.
    """
    if not results:
        raise ValueError("No DRT results provided to plot.")

    fig, ax = plt.subplots(figsize=(8, 5))

    for label, result in results:
        data = result.get_drt_data()
        if len(data) == 3:
            tau, gamma_re, gamma_im = data
            freq = 1.0 / (2.0 * np.pi * tau)
            ax.semilogx(freq, gamma_re, linewidth=1.5, label=f"{label} (Re)")
            ax.semilogx(freq, gamma_im, linewidth=1.5, linestyle="--", label=f"{label} (Im)")
        else:
            tau, gamma = data
            freq = 1.0 / (2.0 * np.pi * tau)
            ax.semilogx(freq, gamma, linewidth=1.5, label=label)

    # High frequency on the left, low frequency on the right, to match the
    # Nyquist plot's orientation.
    ax.invert_xaxis()

    ax.set_xlabel("Frequency (Hz)", fontsize=10)
    ax.set_ylabel(r"$\gamma$ ($\Omega$)", fontsize=10)
    ax.set_title(title, fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(fontsize=9, loc="best")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to: {save_path}")

    if show:
        plt.show()

    return fig, ax


GID_KEPT_POINTS = "kept_points"
GID_REMOVED_POINTS = "removed_points"

# Attribute name used to stash per-point (dataset label, frequency) metadata
# on plotted artists, so GUI code can recover "what set / what freq" on click
# without re-deriving it from the raw x/y data.
POINT_META_ATTR = "eis_point_meta"


def _attach_point_meta(artist, label: str, freq) -> None:
    """Stash the dataset label and per-point frequencies on artist for later
    lookup (e.g. by a click handler), keyed by the point's index in the
    plotted array."""
    setattr(artist, POINT_META_ATTR, {"label": label, "freq": freq})


def _plot_series(ax: Axes, Z, freq, label: str, style: str) -> None:
    """Draw one dataset's impedance on ax, either as a scatter or a connected line."""
    x, y = Z.real, -Z.imag
    if style == "scatter":
        artist = ax.scatter(x, y, s=15, label=label, picker=True, pickradius=5)
    elif style == "line":
        (artist,) = ax.plot(x, y, linewidth=1, label=label, picker=5)
    else:
        raise ValueError(f"Unknown style '{style}'; expected 'scatter' or 'line'.")
    artist.set_gid(GID_KEPT_POINTS)
    _attach_point_meta(artist, label, freq)


def _plot_removed(ax: Axes, Z, freq, label: str, legend_label: Optional[str]) -> None:
    """Draw masked-out points as muted 'x' markers."""
    if Z.size == 0:
        return
    x, y = Z.real, -Z.imag
    artist = ax.scatter(
        x, y, s=25, marker="x", color="0.6", linewidths=1, label=legend_label,
        zorder=2, picker=True, pickradius=5,
    )
    artist.set_gid(GID_REMOVED_POINTS)
    _attach_point_meta(artist, f"{label} (removed)", freq)


def equal_aspect_limits(
    xmin: float, xmax: float, ymin: float, ymax: float, *, include_origin: bool = True
) -> Tuple[float, float, float, float]:
    """Pad (xmin, xmax, ymin, ymax) so both axes span the same range (for
    aspect='equal'). When include_origin is True (the Nyquist plot default),
    the origin is folded into the range first so 0 is never cropped out."""
    if include_origin:
        xmin, xmax = min(xmin, 0), max(xmax, 0)
        ymin, ymax = min(ymin, 0), max(ymax, 0)

    # Pad the narrower axis so both spans match, keeping the scale equal.
    span = max(xmax - xmin, ymax - ymin)
    x_pad = (span - (xmax - xmin)) / 2
    y_pad = (span - (ymax - ymin)) / 2
    return xmin - x_pad, xmax + x_pad, ymin - y_pad, ymax + y_pad


def _format_ax(ax: Axes) -> None:
    """Apply standard Nyquist formatting to an axes."""
    ax.set_xlabel("Z' (Ω)", fontsize=10)
    ax.set_ylabel("-Z'' (Ω)", fontsize=10)
    ax.grid(True, linestyle="--", alpha=0.5)

    xlo, xhi, ylo, yhi = equal_aspect_limits(*ax.get_xlim(), *ax.get_ylim())
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    ax.set_aspect("equal", adjustable="box")

    ax.axhline(0, color=ORIGIN_COLOR, linewidth=ORIGIN_WIDTH, zorder=1)
    ax.axvline(0, color=ORIGIN_COLOR, linewidth=ORIGIN_WIDTH, zorder=1)

    ax.legend(
        fontsize=9,
        loc="upper left",
        bbox_to_anchor=(1.05, 1),
        borderaxespad=0,
    )