"""Publication figures generated only from current-run outputs."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from .metrics import (
    ALL_METRICS,
    lead_lag_correlation,
    peak_months,
    spectrum_period,
)


OBS = "#222222"
LINEAR = "#ef8a62"
BASE = "#ca0020"
MEMORY = "#2166ac"
LOW = "#4393c3"
HIGH = "#d6604d"
MID = "#333333"
SCORE_CMAP = LinearSegmentedColormap.from_list(
    "scoremap", ["#b2182b", "#f7f7f7", "#2166ac"]
)

MODEL_ORDER = [
    "Observed",
    "AR(1)",
    "Seasonal AR(1)",
    "Seasonal AR(2)",
    "Seasonal recharge VAR",
    "Linear Jin RO",
    "Jin++ (memory off)",
    "Jin++ (memory on)",
]

METRIC_LABELS = [
    "Skew",
    "Kurt.",
    "r6",
    "r12",
    "r24",
    "Period",
    "Band",
    "Spectrum",
    "Winter",
    "R",
    "Warm/cold",
    "E→L",
    "L→L",
    "L→E",
    "H leads",
]


def set_figure_style(*, font_size: float = 10.0) -> None:
    """Apply the publication defaults while leaving them editable in notebooks."""
    plt.rcParams.update(
        {
            "font.size": font_size,
            "axes.labelsize": font_size,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.hashsalt": "corrected-jinpp",
        }
    )


def _short(name: str) -> str:
    return {
        "Seasonal AR(1)": "Seasonal AR",
        "Seasonal recharge VAR": "Seasonal VAR",
        "Linear Jin RO": "Jin",
        "Jin++ (memory off)": "Jin++",
        "Jin++ (memory on)": "Jin++ memory",
    }.get(name, name)


def save_figure(
    fig: plt.Figure,
    directory: str | Path,
    name: str,
    *,
    formats: tuple[str, ...] = ("pdf", "svg"),
    dpi: int = 180,
    close: bool = False,
) -> list[Path]:
    """Export one already-created figure without hiding it from the notebook.

    This is intentionally separate from figure construction: users can change
    axes, colours, labels, limits, or dimensions before exporting.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for extension in formats:
        path = directory / f"{name}.{extension}"
        metadata = {"CreationDate": None} if extension == "pdf" else {"Date": None}
        fig.savefig(path, bbox_inches="tight", dpi=dpi, metadata=metadata)
        written.append(path)
    if close:
        plt.close(fig)
    return written


def _finish(fig: plt.Figure, directory: Path | None, name: str) -> plt.Figure:
    """Optionally export a figure, then return it for interactive inspection."""
    if directory is not None:
        save_figure(fig, directory, name, close=True)
    return fig


def _series(
    observations: pd.DataFrame,
    models: Mapping[str, pd.DataFrame],
    name: str,
    variable: str,
) -> np.ndarray:
    frame = observations if name == "Observed" else models[name]
    return frame[variable].to_numpy(float)


def plot_schematic(directory: Path | None = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12.0, 5.4))
    ax.set_axis_off()
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1)

    def box(x, y, w, h, text, fc="#f7f7f7", fontsize=9):
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.025",
            fc=fc,
            ec="0.3",
            lw=1.0,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)
        return x, y, w, h

    def arrow(first, second, style="-", color="0.25"):
        x1, y1, w1, h1 = first
        x2, y2, _, h2 = second
        ax.add_patch(
            FancyArrowPatch(
                (x1 + w1, y1 + h1 / 2),
                (x2, y2 + h2 / 2),
                arrowstyle="-|>",
                mutation_scale=10,
                lw=1.0,
                color=color,
                linestyle=style,
            )
        )

    headings = [
        (0.10, "Parent budgets"),
        (0.35, "Projected variables"),
        (0.60, "Physical closures"),
        (0.815, "Coupled model"),
        (0.965, "Tests and limit"),
    ]
    for x, text in headings:
        ax.text(x, 0.96, text, ha="center", va="top", fontsize=10, weight="bold")
    ys = [0.78, 0.58, 0.38, 0.18]
    parents = [
        box(0.02, ys[0], 0.18, 0.10, "mixed-layer\nheat budget", "#fddbc7"),
        box(0.02, ys[1], 0.18, 0.10, "equatorial\nupper ocean", "#fddbc7"),
        box(0.02, ys[2], 0.18, 0.10, "off-equatorial\nmemory budget", "#fddbc7"),
        box(0.02, ys[3], 0.18, 0.10, "weather-scale\nforcing", "#fddbc7"),
    ]
    variables = [
        box(0.27, ys[0], 0.18, 0.10, "$T_E,T_C$\nSST modes"),
        box(0.27, ys[1], 0.18, 0.10, "$H,K$\nrecharge and tilt"),
        box(0.27, ys[2], 0.18, 0.10, "$H_N,H_S$ and\n$R_1,R_2,R_3,R_K$"),
        box(0.27, ys[3], 0.18, 0.10, "$q,e$\nwind bursts"),
    ]
    closures = [
        box(0.52, ys[0], 0.18, 0.10, "nonlinear\n$\\Phi(h)$ and SST damping", "#d1e5f0"),
        box(0.52, ys[1], 0.18, 0.10, "wind-stress\nclosure $\\tau$", "#d1e5f0"),
        box(0.52, ys[2], 0.18, 0.10, "delayed wind-forced\nmemory proxy", "#d1e5f0"),
        box(0.52, ys[3], 0.18, 0.10, "state-dependent\njump rates", "#d1e5f0"),
    ]
    simulator = box(0.76, 0.25, 0.11, 0.55, "Jin++\nENSO\nsimulator", "#e0e0e0", 10)
    score = box(0.92, 0.58, 0.11, 0.12, "moment\nscorecard", "#e0e0e0")
    limit = box(0.92, 0.30, 0.11, 0.12, "Jin oscillator\nlinear limit", "#e0e0e0")
    for first, second, third in zip(parents, variables, closures):
        arrow(first, second)
        arrow(second, third)
        x1, y1, w1, h1 = third
        ax.add_patch(
            FancyArrowPatch(
                (x1 + w1, y1 + h1 / 2),
                (simulator[0], y1 + h1 / 2),
                arrowstyle="-|>",
                mutation_scale=10,
                lw=1.0,
                color="0.25",
            )
        )
    arrow(simulator, score)
    arrow(simulator, limit, style="--", color="0.35")
    ax.text(
        0.52,
        0.04,
        "The memory chain is a temporal proxy; it does not resolve curl, propagation or reflection.",
        ha="center",
        fontsize=8,
    )
    return _finish(fig, directory, "fig_schematic")


def plot_scorecard(
    errors: pd.DataFrame,
    scores: pd.DataFrame,
    directory: Path | None = None,
) -> plt.Figure:
    error_map = errors.set_index("model").reindex(MODEL_ORDER)
    matrix = error_map[ALL_METRICS].to_numpy(float)
    plot_matrix = np.clip(matrix, 0.0, 3.0)
    score_map = scores.set_index("model")["core_rms_score"]
    score_values = score_map.reindex(MODEL_ORDER).to_numpy(float)
    fig = plt.figure(figsize=(10.0, 4.5))
    grid = GridSpec(1, 3, width_ratios=[15, 1.8, 0.55], wspace=0.18)
    ax = fig.add_subplot(grid[0, 0])
    image = ax.imshow(plot_matrix, aspect="auto", cmap=SCORE_CMAP, vmin=0, vmax=3)
    ax.set_xticks(np.arange(len(METRIC_LABELS)))
    ax.set_xticklabels(METRIC_LABELS, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(MODEL_ORDER)))
    ax.set_yticklabels([_short(name) for name in MODEL_ORDER])
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            if np.isfinite(value):
                colour = "white" if plot_matrix[i, j] < 0.25 or plot_matrix[i, j] > 2.2 else "black"
                label = f"{value:.2f}" if ALL_METRICS[j] == "period" else f"{value:.1f}"
                ax.text(j, i, label, ha="center", va="center", fontsize=8.5, color=colour)
            else:
                ax.text(j, i, "—", ha="center", va="center", fontsize=8.5, color="0.4")
    ax2 = fig.add_subplot(grid[0, 1], sharey=ax)
    ax2.barh(np.arange(len(MODEL_ORDER)), score_values, color="#b2182b")
    ax2.set_xlabel("Core RMS")
    ax2.tick_params(axis="y", left=False, labelleft=False)
    ax2.set_xlim(0, max(np.nanmax(score_values) * 1.25, 1.0))
    for i, value in enumerate(score_values):
        ax2.text(value + 0.04, i, f"{value:.2f}", va="center", fontsize=8.5)
    colour_axis = fig.add_subplot(grid[0, 2])
    fig.colorbar(image, cax=colour_axis).set_label("Normalised absolute error")
    return _finish(fig, directory, "fig_scorecard")


def plot_spectrum(observations, models, directory: Path | None = None) -> plt.Figure:
    selected = [
        ("Observed", OBS),
        ("Linear Jin RO", LINEAR),
        ("Jin++ (memory off)", BASE),
        ("Jin++ (memory on)", MEMORY),
    ]
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    for name, colour in selected:
        x = _series(observations, models, name, "nino34")
        period, density = spectrum_period(x)
        use = (period >= 1.0) & (period <= 10.0)
        p = period[use]
        d = density[use]
        d /= np.trapezoid(d, p)
        ax.plot(p, d, lw=2.0, color=colour, label=_short(name))
    ax.set_xlabel("Period (years)")
    ax.set_ylabel("Normalised density per year of period")
    ax.legend(frameon=False)
    return _finish(fig, directory, "fig_spectrum")


def plot_timelines(observations, models, directory: Path | None = None) -> plt.Figure:
    length = len(observations)
    selections = [
        ("Observed", observations["nino34"].to_numpy(float)),
        ("Linear Jin RO", models["Linear Jin RO"]["nino34"].to_numpy(float)[:length]),
        ("Jin++ (memory off)", models["Jin++ (memory off)"]["nino34"].to_numpy(float)[1200 : 1200 + length]),
        ("Jin++ (memory on)", models["Jin++ (memory on)"]["nino34"].to_numpy(float)[1800 : 1800 + length]),
    ]
    fig, axes = plt.subplots(4, 1, figsize=(8.3, 7.1), sharex=True)
    for ax, (name, x) in zip(axes, selections):
        years = np.arange(len(x)) / 12.0
        ax.axhline(0, color="0.7", lw=0.8)
        ax.plot(years, x, color="0.2", lw=1.2)
        ax.fill_between(years, 0, x, where=x >= 0, color=HIGH, alpha=0.8, linewidth=0)
        ax.fill_between(years, 0, x, where=x < 0, color=LOW, alpha=0.8, linewidth=0)
        ax.text(0.01, 0.86, _short(name), transform=ax.transAxes, va="top", fontsize=11)
        ax.set_ylabel("°C")
    axes[-1].set_xlabel("Years from segment start")
    fig.tight_layout(h_pad=0.3)
    return _finish(fig, directory, "fig_timelines")


def plot_phase_space(observations, models, directory: Path | None = None) -> plt.Figure:
    length = len(observations)
    sets = [
        ("Observed", observations["nino34"].to_numpy(), observations["heat"].to_numpy(), OBS),
        ("Linear Jin RO", models["Linear Jin RO"]["nino34"].to_numpy()[:length], models["Linear Jin RO"]["heat"].to_numpy()[:length], LINEAR),
        ("Jin++ (memory off)", models["Jin++ (memory off)"]["nino34"].to_numpy()[1200 : 1200 + length], models["Jin++ (memory off)"]["heat"].to_numpy()[1200 : 1200 + length], BASE),
        ("Jin++ (memory on)", models["Jin++ (memory on)"]["nino34"].to_numpy()[1800 : 1800 + length], models["Jin++ (memory on)"]["heat"].to_numpy()[1800 : 1800 + length], MEMORY),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 7.0), sharex=True, sharey=True)
    for ax, (name, x, h, colour) in zip(axes.flat, sets):
        xx = (x - np.mean(x)) / np.std(x)
        hh = (h - np.mean(h)) / np.std(h)
        ax.plot(hh, xx, color=colour, lw=1.2)
        ax.axhline(0, color="0.85", lw=0.8)
        ax.axvline(0, color="0.85", lw=0.8)
        ax.text(0.03, 0.95, _short(name), transform=ax.transAxes, va="top", fontsize=11)
    for ax in axes[:, 0]:
        ax.set_ylabel("Standardised Niño-3.4")
    for ax in axes[-1, :]:
        ax.set_xlabel("Standardised heat content")
    fig.tight_layout()
    return _finish(fig, directory, "fig_phase_space")


def plot_transitions(statistics, directory: Path | None = None) -> plt.Figure:
    rows = statistics.set_index("model")
    names = ["Observed", "Linear Jin RO", "Jin++ (memory off)", "Jin++ (memory on)"]
    colours = [OBS, LINEAR, BASE, MEMORY]
    metrics = ["E_to_E", "E_to_L", "L_to_L", "L_to_E"]
    values = rows.loc[names, metrics].to_numpy(float)
    fig, ax = plt.subplots(figsize=(6.8, 4.3))
    width = 0.18
    positions = np.arange(len(metrics))
    for i, (name, colour) in enumerate(zip(names, colours)):
        x = positions + (i - 1.5) * width
        ax.bar(x, values[i], width=width, color=colour, label=_short(name))
        for xx, value in zip(x, values[i]):
            ax.text(xx, value + 0.025, f"{value:.2f}", ha="center", fontsize=8)
    ax.set_xticks(positions)
    ax.set_xticklabels(["E→E", "E→L", "L→L", "L→E"])
    ax.set_ylabel("Probability")
    ax.set_ylim(0, max(0.86, np.nanmax(values) + 0.15))
    ax.legend(frameon=False, ncol=2)
    return _finish(fig, directory, "fig_transitions")


def plot_error_heatmap(
    errors,
    scores,
    directory: Path | None = None,
    name: str = "fig_errors",
) -> plt.Figure:
    order = errors["model"].tolist()
    matrix = errors.set_index("model").loc[order, ALL_METRICS].to_numpy(float)
    plot_matrix = np.clip(matrix, 0, 3)
    score_values = scores.set_index("model").loc[order, "core_rms_score"].to_numpy(float)
    fig = plt.figure(figsize=(10.0, max(4.2, 0.42 * len(order))))
    grid = GridSpec(1, 3, width_ratios=[15, 1.8, 0.55], wspace=0.18)
    ax = fig.add_subplot(grid[0, 0])
    image = ax.imshow(plot_matrix, aspect="auto", cmap=SCORE_CMAP, vmin=0, vmax=3)
    ax.set_xticks(np.arange(len(METRIC_LABELS)))
    ax.set_xticklabels(METRIC_LABELS, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels(order)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            colour = "white" if plot_matrix[i, j] < 0.25 or plot_matrix[i, j] > 2.2 else "black"
            label = f"{matrix[i, j]:.2f}" if ALL_METRICS[j] == "period" else f"{matrix[i, j]:.1f}"
            ax.text(j, i, label, ha="center", va="center", fontsize=8, color=colour)
    ax2 = fig.add_subplot(grid[0, 1], sharey=ax)
    ax2.barh(np.arange(len(order)), score_values, color="#b2182b")
    ax2.set_xlabel("Core RMS")
    ax2.tick_params(axis="y", left=False, labelleft=False)
    ax2.set_xlim(0, max(np.nanmax(score_values) * 1.25, 1.0))
    for i, value in enumerate(score_values):
        ax2.text(value + 0.04, i, f"{value:.2f}", va="center", fontsize=8.5)
    cax = fig.add_subplot(grid[0, 2])
    fig.colorbar(image, cax=cax).set_label("Normalised absolute error")
    return _finish(fig, directory, name)


def plot_memory_sensitivity(sensitivity, directory: Path | None = None) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.8, 4.5))
    ax.plot(sensitivity["memory_factor"], sensitivity["core_rms_score"], marker="o", lw=2, color=MID, label="core RMS")
    ax.plot(sensitivity["memory_factor"], sensitivity["L_to_L_error"] / 0.15, marker="o", lw=2, color=BASE, label="L→L error")
    ax.plot(sensitivity["memory_factor"], sensitivity["ac12_error"] / 0.12, marker="o", lw=2, color=MEMORY, label="r12 error")
    ax.plot(sensitivity["memory_factor"], sensitivity["spectrum_error"] / 0.45, marker="o", lw=2, color="#4d9221", label="spectrum error")
    ax.set_xlabel("Delayed-memory source multiplier")
    ax.set_ylabel("Normalised error")
    ax.legend(frameon=False, ncol=2)
    return _finish(fig, directory, "fig_memory_sensitivity")


def plot_leadlag(observations, models, directory: Path | None = None) -> plt.Figure:
    selected = [
        ("Observed", OBS),
        ("Linear Jin RO", LINEAR),
        ("Jin++ (memory off)", BASE),
        ("Jin++ (memory on)", MEMORY),
    ]
    fig, ax = plt.subplots(figsize=(6.8, 4.5))
    for name, colour in selected:
        x = _series(observations, models, name, "nino34")
        h = _series(observations, models, name, "heat")
        lags, correlation = lead_lag_correlation(h, x)
        ax.plot(lags, correlation, lw=2, color=colour, label=_short(name))
    ax.axhline(0, color="0.8", lw=0.8)
    ax.axvline(0, color="0.8", lw=0.8)
    ax.set_xlabel("Lag (months; positive means heat content leads SST)")
    ax.set_ylabel("Correlation")
    ax.set_xlim(-18, 18)
    ax.legend(frameon=False)
    return _finish(fig, directory, "fig_leadlag")


def plot_peakmonths(observations, models, directory: Path | None = None) -> plt.Figure:
    selected = [
        ("Observed", OBS),
        ("Linear Jin RO", LINEAR),
        ("Jin++ (memory off)", BASE),
        ("Jin++ (memory on)", MEMORY),
    ]
    centres = np.arange(1, 13)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    for name, colour in selected:
        months = peak_months(_series(observations, models, name, "nino34"))
        counts = np.array([(months == month).sum() for month in centres], float)
        fraction = counts / counts.sum()
        ax.plot(centres, fraction, marker="o", lw=2, color=colour, label=_short(name))
    ax.set_xticks(centres)
    ax.set_xticklabels(list("JFMAMJJASOND"))
    ax.set_ylabel("Fraction of warm peaks")
    ax.set_xlabel("Calendar month")
    ax.legend(frameon=False, ncol=2)
    return _finish(fig, directory, "fig_peakmonths")


def plot_parameter_perturbations(
    perturbations,
    directory: Path | None = None,
) -> plt.Figure:
    experiments = perturbations["experiment"].drop_duplicates().tolist()
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.1), sharex=True, sharey=True)
    styles = {0.7: (LOW, 1.6), 1.0: (MID, 2.1), 1.3: (HIGH, 1.6)}
    for ax, experiment in zip(axes.flat, experiments):
        subset = perturbations[perturbations["experiment"] == experiment]
        for factor, (colour, width) in styles.items():
            curve = subset[np.isclose(subset["factor"], factor)]
            ax.plot(curve["year"], curve["nino34"], color=colour, lw=width, label=f"{factor:.1f}×")
        ax.axhline(0, color="0.8", lw=0.8)
        ax.text(0.03, 0.92, experiment, transform=ax.transAxes, va="top")
    for ax in axes[:, 0]:
        ax.set_ylabel("Niño-3.4-like anomaly (°C)")
    for ax in axes[-1, :]:
        ax.set_xlabel("Years after common initial state")
    axes[0, 0].legend(frameon=False, ncol=3, loc="lower left")
    fig.tight_layout()
    return _finish(fig, directory, "fig_parameter_perturbations")


def generate_figures(
    *,
    observations: pd.DataFrame,
    models: Mapping[str, pd.DataFrame],
    statistics: pd.DataFrame,
    errors: pd.DataFrame,
    scores: pd.DataFrame,
    ablation_errors: pd.DataFrame,
    ablation_scores: pd.DataFrame,
    sensitivity: pd.DataFrame,
    perturbations: pd.DataFrame,
    directory: Path,
) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    set_figure_style()
    plot_schematic(directory)
    plot_scorecard(errors, scores, directory)
    plot_spectrum(observations, models, directory)
    plot_timelines(observations, models, directory)
    plot_phase_space(observations, models, directory)
    plot_transitions(statistics, directory)
    plot_error_heatmap(ablation_errors, ablation_scores, directory, "fig_ablations")
    plot_memory_sensitivity(sensitivity, directory)
    plot_leadlag(observations, models, directory)
    plot_peakmonths(observations, models, directory)
    plot_parameter_perturbations(perturbations, directory)
    return sorted(directory.glob("fig_*.pdf"))
