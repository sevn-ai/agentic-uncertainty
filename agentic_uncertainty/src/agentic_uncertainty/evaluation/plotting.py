"""Visualization utilities for uncertainty evaluation.

Provides ROC curves, calibration plots, and confidence histograms.
Clean, publication-quality figures with minimal chrome.
"""

import io
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from numpy.typing import ArrayLike
from PIL import Image
from sklearn.metrics import roc_curve

# Path to logo assets
ASSETS_DIR = Path(__file__).parent.parent.parent.parent / "assets"

# Flag to track if style setup has been done
_STYLE_SETUP_DONE = False


def _load_logo(model: str, size: int = 50) -> np.ndarray | None:
    """Load and resize a model logo for embedding in plots.

    Args:
        model: Model name ("gpt", "gemini", "claude").
        size: Target size in pixels.

    Returns:
        RGBA numpy array or None if loading fails.
    """
    logo_bases = {
        "gpt": "openai",
        "gemini": "gemini",
        "claude": "anthropic",
    }

    if model not in logo_bases:
        return None

    base_name = logo_bases[model]

    # Prefer pre-converted PNG files
    png_path = ASSETS_DIR / f"{base_name}.png"
    if png_path.exists():
        try:
            img = Image.open(png_path)
            # Convert to RGBA for consistent handling
            img = img.convert("RGBA")
            img = img.resize((size, size), Image.Resampling.LANCZOS)
            return np.array(img)
        except Exception:
            pass

    # Fallback: try SVG conversion if cairosvg is available
    svg_path = ASSETS_DIR / f"{base_name}.svg"
    if svg_path.exists():
        try:
            import cairosvg

            png_data = cairosvg.svg2png(
                url=str(svg_path),
                output_width=size,
                output_height=size,
            )
            img = Image.open(io.BytesIO(png_data))
            img = img.convert("RGBA")
            return np.array(img)
        except ImportError:
            pass
        except Exception:
            pass

    return None

# =============================================================================
# Style Configuration (swe-self-play aesthetic)
# =============================================================================

# Color palette (matches LaTeX document colors)
COLORS = {
    "pre_execution": "#3282B4",    # Cerulean blue (PreExecBlue: 50,130,180)
    "post_execution": "#5A8C6E",   # Sage green (PostExecGreen: 90,140,110)
    "adversarial": "#C8553C",      # Coral (AdvCoral: 200,85,60)
    "gpt": "#2171B5",              # Blue (OpenAI)
    "gemini": "#8B5CF6",           # Purple (Gemini brand)
    "claude": "#E8835F",           # Orange/coral (Anthropic brand)
    "reference": "#CCCCCC",        # Light gray
    "success": "#2ca02c",          # Green (for histograms)
    "failure": "#d62728",          # Red (for histograms)
}

# Marker styles - smaller, cleaner, with subtle edges
MARKERS = {
    "pre_execution": {"marker": "o", "markersize": 6, "markeredgecolor": "white", "markeredgewidth": 0.5},
    "post_execution": {"marker": "s", "markersize": 6, "markeredgecolor": "white", "markeredgewidth": 0.5},
    "adversarial": {"marker": "^", "markersize": 7, "markeredgecolor": "white", "markeredgewidth": 0.5},
    "gpt": {"marker": "o", "markersize": 6, "markeredgecolor": "white", "markeredgewidth": 0.5},
    "gemini": {"marker": "s", "markersize": 6, "markeredgecolor": "white", "markeredgewidth": 0.5},
    "claude": {"marker": "D", "markersize": 5, "markeredgecolor": "white", "markeredgewidth": 0.5},
}

# Method labels (short, clean)
METHOD_LABELS = {
    "exploration_direct": "Pre-Exec",
    "review_direct": "Post-Exec",
    "review_adversarial": "Adv. Post-Exec",
    "ensemble_average": "Ensemble Avg",
    "ensemble_min": "Ensemble Min",
    "ensemble_max": "Ensemble Max",
}

# Model display names
MODEL_LABELS = {
    "gpt-5.2-codex": "GPT-5.2-Codex",
    "gemini-3-pro-preview": "Gemini-3-Pro",
    "claude-opus-4-5": "Claude Opus 4.5",
}

# Consistent line width
LINE_WIDTH = 2.5

# Figure sizes
FIGSIZE_SINGLE = (4.0, 3.5)
FIGSIZE_WIDE = (5.5, 3.5)
FIGSIZE_MULTI = (12.0, 3.5)

# Font sizes for publication-quality figures
FONTSIZE = {
    "title": 16,           # Panel/figure titles
    "title_with_logo": 15, # Slightly smaller when logo present
    "axis_label": 14,      # X and Y axis labels
    "tick_label": 12,      # Tick mark labels
    "legend": 12,          # Legend text
    "annotation": 12,      # In-plot annotations (e.g., "Success", "Failure")
    "reference_text": 10,  # Reference line labels (e.g., "chance", "perfect")
}

# Logo zoom size
LOGO_ZOOM = 0.12


def setup_latex_style() -> None:
    """Configure matplotlib for clean, publication-quality figures.

    Uses sans-serif fonts, minimal chrome (no top/right spines),
    and thick lines matching the swe-self-play aesthetic.
    """
    global _STYLE_SETUP_DONE
    if _STYLE_SETUP_DONE:
        return

    base_settings = {
        # Disable external LaTeX - use built-in mathtext instead
        "text.usetex": False,
        "mathtext.fontset": "dejavusans",

        # Font settings - sans-serif for clean look
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial", "sans-serif"],
        "font.size": 13,

        # Axes - no top/right spines, no default grid
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "axes.linewidth": 0.8,
        "axes.grid": False,  # Add grid manually via helper
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,

        # Ticks
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,

        # Legend - clean, minimal
        "legend.fontsize": 11,
        "legend.framealpha": 0.95,
        "legend.edgecolor": "none",
        "legend.fancybox": False,

        # Figure
        "figure.figsize": FIGSIZE_SINGLE,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,

        # Lines - thick for visibility
        "lines.linewidth": LINE_WIDTH,
        "lines.markersize": 8,

        # Grid - subtle when added
        "grid.alpha": 0.4,
        "grid.linewidth": 0.5,
        "grid.color": "#E5E5E5",
    }

    plt.rcParams.update(base_settings)
    _STYLE_SETUP_DONE = True


def add_subtle_grid(ax: plt.Axes, axis: str = "y") -> None:
    """Add subtle horizontal grid lines.

    Args:
        ax: Matplotlib axes.
        axis: Which axis to add grid lines ("y", "x", or "both").
    """
    ax.grid(True, axis=axis, linewidth=0.5, color="#E5E5E5", zorder=0)


def _save_figure(fig: plt.Figure, save_path: Path | str) -> None:
    """Save figure as both PDF (vector) and PNG (raster).

    Args:
        fig: Matplotlib figure.
        save_path: Path to save (extension will be replaced).
    """
    save_path = Path(save_path)
    base_path = save_path.parent / save_path.stem

    # Ensure directory exists
    base_path.parent.mkdir(parents=True, exist_ok=True)

    # Save as PDF (publication quality, vector)
    fig.savefig(f"{base_path}.pdf", dpi=300, bbox_inches="tight")

    # Save as PNG (preview, raster)
    fig.savefig(f"{base_path}.png", dpi=150, bbox_inches="tight")


def add_endpoint_label(
    ax: plt.Axes,
    x: float,
    y: float,
    label: str,
    color: str,
    offset: tuple[int, int] = (5, 0),
    fontsize: int = 9,
) -> None:
    """Add data label at line endpoint.

    Args:
        ax: Matplotlib axes.
        x: X coordinate.
        y: Y coordinate.
        label: Text to display.
        color: Text color.
        offset: Offset in points (x, y).
        fontsize: Font size.
    """
    ax.annotate(
        label,
        xy=(x, y),
        xytext=offset,
        textcoords="offset points",
        fontsize=fontsize,
        color=color,
        fontweight="bold",
        va="center",
    )


def _get_method_style(method: str) -> dict:
    """Get color, marker, and line style for a method.

    Args:
        method: Method name (e.g., "exploration_direct").

    Returns:
        Dict with color, marker, and line style kwargs.
    """
    # Map method to style key
    if "exploration" in method or "pre" in method.lower():
        key = "pre_execution"
        linestyle = "-"
        linewidth = 3.0
    elif "adversarial" in method:
        key = "adversarial"
        linestyle = ":"
        linewidth = 4.0  # Thicker for dotted to be visible
    elif "review" in method or "post" in method.lower():
        key = "post_execution"
        linestyle = "--"
        linewidth = 3.0
    else:
        key = "pre_execution"  # Default
        linestyle = "-"
        linewidth = 3.0

    return {
        "color": COLORS.get(key, "#666666"),
        "linestyle": linestyle,
        "linewidth": linewidth,
        **MARKERS.get(key, {"marker": "o", "markersize": 8}),
    }


def _get_model_style(model: str) -> dict:
    """Get color and marker style for a model.

    Args:
        model: Model name.

    Returns:
        Dict with color and marker kwargs.
    """
    model_lower = model.lower()
    if "gpt" in model_lower:
        key = "gpt"
    elif "gemini" in model_lower:
        key = "gemini"
    elif "claude" in model_lower:
        key = "claude"
    else:
        key = "gpt"  # Default

    return {
        "color": COLORS.get(key, "#666666"),
        **MARKERS.get(key, {"marker": "o", "markersize": 8}),
    }


def _format_method_name(name: str) -> str:
    """Format method name for display.

    Uses short labels from METHOD_LABELS if available,
    otherwise converts underscores to spaces.
    """
    if name in METHOD_LABELS:
        return METHOD_LABELS[name]
    # Replace underscores with spaces for readability
    formatted = name.replace("_", " ")
    # Capitalize words
    formatted = formatted.title()
    return formatted


def _format_model_name(name: str) -> str:
    """Format model name for display.

    Uses clean labels from MODEL_LABELS if available.
    """
    if name in MODEL_LABELS:
        return MODEL_LABELS[name]
    # Try lowercase match (both directions for partial names)
    name_lower = name.lower()
    for key, label in MODEL_LABELS.items():
        if key in name_lower or name_lower in key:
            return label
    # Fallback: clean up the name
    return name.replace("-", " ").replace("_", " ").title()


def plot_roc_curve(
    predictions: ArrayLike,
    labels: ArrayLike,
    title: str | None = None,
    save_path: Path | None = None,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot ROC curve with AUC annotation.

    Args:
        predictions: Predicted probabilities in [0, 1].
        labels: Binary labels (0 or 1).
        title: Plot title (None for no title).
        save_path: Optional path to save figure.
        ax: Optional axes to plot on.

    Returns:
        Matplotlib figure.
    """
    setup_latex_style()

    predictions = np.asarray(predictions)
    labels = np.asarray(labels)

    fpr, tpr, _ = roc_curve(labels, predictions)

    # Compute AUC
    from sklearn.metrics import roc_auc_score

    auc = roc_auc_score(labels, predictions)

    if ax is None:
        fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    else:
        fig = ax.get_figure()

    style = _get_method_style("pre_execution")  # Default style
    ax.plot(
        fpr,
        tpr,
        lw=LINE_WIDTH,
        color=style["color"],
        marker=style["marker"],
        markersize=style["markersize"],
        markeredgecolor=style["markeredgecolor"],
        markeredgewidth=style["markeredgewidth"],
        markevery=max(1, len(fpr) // 8),
        label=f"AUROC = {auc:.3f}",
        zorder=3,
    )

    # Reference line (chance)
    ax.plot([0, 1], [0, 1], color=COLORS["reference"], linestyle="-", lw=1, zorder=1)
    ax.text(0.55, 0.48, "chance", fontsize=8, color="#888888", va="top")

    # Add subtle grid
    add_subtle_grid(ax, axis="both")

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    if title:
        ax.set_title(title, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)

    plt.tight_layout()

    if save_path:
        _save_figure(fig, save_path)

    return fig


def plot_roc_curves_comparison(
    results: dict[str, tuple[ArrayLike, ArrayLike]],
    title: str | None = None,
    save_path: Path | None = None,
    methods_to_show: list[str] | None = None,
) -> plt.Figure:
    """Plot multiple ROC curves on same axes for comparison.

    Clean style with curated colors and thick lines. AUROC values in legend only.

    Args:
        results: Dict mapping method name to (predictions, labels) tuple.
        title: Plot title (None for no title - use figure caption instead).
        save_path: Optional path to save figure.
        methods_to_show: Methods to include (default: pre, post, adversarial).

    Returns:
        Matplotlib figure.
    """
    setup_latex_style()
    from sklearn.metrics import roc_auc_score

    # Default to showing only 3 key methods
    if methods_to_show is None:
        methods_to_show = ["exploration_direct", "review_direct", "review_adversarial"]

    # Filter results to only requested methods
    filtered_results = {k: v for k, v in results.items() if k in methods_to_show}
    if not filtered_results:
        filtered_results = results  # Fallback to all if none match

    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)

    # Sort by AUROC for consistent legend ordering (best first)
    sorted_methods = []
    for method, (preds, labels) in filtered_results.items():
        preds = np.asarray(preds)
        labels = np.asarray(labels)
        auc = roc_auc_score(labels, preds)
        sorted_methods.append((method, preds, labels, auc))
    sorted_methods.sort(key=lambda x: -x[3])  # Descending by AUC

    # Plot each method with curated style
    for method, preds, labels, auc in sorted_methods:
        fpr, tpr, _ = roc_curve(labels, preds)
        style = _get_method_style(method)

        ax.plot(
            fpr,
            tpr,
            lw=LINE_WIDTH,
            color=style["color"],
            marker=style["marker"],
            markersize=style["markersize"],
            markeredgecolor=style["markeredgecolor"],
            markeredgewidth=style["markeredgewidth"],
            markevery=max(1, len(fpr) // 6),  # Sparse markers
            label=f"{_format_method_name(method)} ({auc:.2f})",
            zorder=3,
        )
        # No endpoint labels - AUROC is in the legend

    # Reference line (chance)
    ax.plot([0, 1], [0, 1], color=COLORS["reference"], linestyle="--", lw=1.2, zorder=1)
    ax.text(0.52, 0.42, "chance", fontsize=9, color="#999999", va="top", style="italic")

    # Add subtle grid
    add_subtle_grid(ax, axis="both")

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    if title:
        ax.set_title(title, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)

    plt.tight_layout()

    if save_path:
        _save_figure(fig, save_path)

    return fig


def plot_roc_curves_multi_model(
    model_results: dict[str, dict[str, tuple[ArrayLike, ArrayLike]]],
    save_path: Path | None = None,
    methods_to_show: list[str] | None = None,
) -> plt.Figure:
    """Plot ROC curves with one subplot per model for clarity.

    Clean style with 3 methods per panel, curated colors, consistent ordering.

    Args:
        model_results: model -> method -> (predictions, labels)
        save_path: Optional path to save figure.
        methods_to_show: Methods to include (default: pre, post, adversarial).

    Returns:
        Matplotlib figure.
    """
    setup_latex_style()
    from sklearn.metrics import roc_auc_score

    # Default to showing only 3 key methods
    if methods_to_show is None:
        methods_to_show = ["exploration_direct", "review_direct", "review_adversarial"]

    # Sort models for consistent ordering (GPT first, then Gemini, then Claude)
    model_order = {"gpt": 0, "gemini": 1, "claude": 2}
    models = sorted(
        model_results.keys(),
        key=lambda m: model_order.get(m.lower().split("-")[0], 99)
    )
    n_models = len(models)

    # Create side-by-side subplots with good spacing
    fig, axes = plt.subplots(1, n_models, figsize=(4.5 * n_models, 4.0))
    if n_models == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        method_data = model_results[model]
        model_label = _format_model_name(model)

        # Filter to requested methods
        filtered_data = {k: v for k, v in method_data.items() if k in methods_to_show}
        if not filtered_data:
            filtered_data = method_data  # Fallback

        # Use consistent method order (same across all panels)
        method_order_list = ["exploration_direct", "review_direct", "review_adversarial"]
        sorted_methods = []
        for method in method_order_list:
            if method in filtered_data:
                preds, labels = filtered_data[method]
                auc = roc_auc_score(np.asarray(labels), np.asarray(preds))
                sorted_methods.append((method, preds, labels, auc))

        for method, preds, labels, auc in sorted_methods:
            preds = np.asarray(preds)
            labels = np.asarray(labels)

            fpr, tpr, _ = roc_curve(labels, preds)
            style = _get_method_style(method)

            ax.plot(
                fpr,
                tpr,
                lw=LINE_WIDTH,
                color=style["color"],
                marker=style["marker"],
                markersize=style["markersize"],
                markeredgecolor=style["markeredgecolor"],
                markeredgewidth=style["markeredgewidth"],
                markevery=max(1, len(fpr) // 6),
                label=f"{_format_method_name(method)} ({auc:.2f})",
                zorder=3,
            )
            # No endpoint labels - AUROC is in the legend

        # Reference line (chance)
        ax.plot([0, 1], [0, 1], color=COLORS["reference"], linestyle="--", lw=1.2, zorder=1)
        ax.text(0.52, 0.42, "chance", fontsize=FONTSIZE["reference_text"], color="#999999", va="top", style="italic")

        # Add subtle grid
        add_subtle_grid(ax, axis="both")

        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel("False Positive Rate", fontsize=FONTSIZE["axis_label"])
        ax.set_ylabel("True Positive Rate", fontsize=FONTSIZE["axis_label"])
        ax.set_title(model_label, fontsize=FONTSIZE["title"], fontweight="bold")
        ax.legend(loc="lower right", fontsize=FONTSIZE["legend"], framealpha=0.95)
        ax.tick_params(labelsize=FONTSIZE["tick_label"])

    plt.tight_layout()

    if save_path:
        _save_figure(fig, save_path)

    return fig


def _compute_calibration_bins(
    predictions: ArrayLike,
    labels: ArrayLike,
    num_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute calibration bin data.

    Returns:
        bin_means: Mean predicted probability per bin.
        bin_accuracies: Actual accuracy per bin.
        bin_counts: Number of samples per bin.
    """
    predictions = np.asarray(predictions)
    labels = np.asarray(labels)

    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    bin_indices = np.digitize(predictions, bin_boundaries[1:-1])

    bin_means = []
    bin_accuracies = []
    bin_counts = []

    for bin_idx in range(num_bins):
        mask = bin_indices == bin_idx
        if np.any(mask):
            bin_means.append(np.mean(predictions[mask]))
            bin_accuracies.append(np.mean(labels[mask]))
            bin_counts.append(np.sum(mask))
        else:
            bin_means.append(np.nan)
            bin_accuracies.append(np.nan)
            bin_counts.append(0)

    return np.array(bin_means), np.array(bin_accuracies), np.array(bin_counts)


def plot_calibration_curve(
    predictions: ArrayLike,
    labels: ArrayLike,
    num_bins: int = 10,
    title: str | None = None,
    save_path: Path | None = None,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot calibration (reliability) diagram.

    Args:
        predictions: Predicted probabilities in [0, 1].
        labels: Binary labels (0 or 1).
        num_bins: Number of bins for calibration.
        title: Plot title (None for no title).
        save_path: Optional path to save figure.
        ax: Optional axes to plot on.

    Returns:
        Matplotlib figure.
    """
    setup_latex_style()

    bin_means, bin_accuracies, bin_counts = _compute_calibration_bins(
        predictions, labels, num_bins
    )

    if ax is None:
        fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
    else:
        fig = ax.get_figure()

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], color=COLORS["reference"], linestyle="-", lw=1.5, zorder=1)
    ax.text(0.85, 0.78, "perfect", fontsize=8, color="#888888", rotation=45, va="bottom")

    # Filter out empty bins
    valid = ~np.isnan(bin_means)
    style = _get_method_style("pre_execution")  # Default style
    ax.plot(
        bin_means[valid],
        bin_accuracies[valid],
        lw=LINE_WIDTH,
        color=style["color"],
        marker=style["marker"],
        markersize=style["markersize"],
        markeredgecolor=style["markeredgecolor"],
        markeredgewidth=style["markeredgewidth"],
        label="Model",
        zorder=3,
    )

    # Add subtle grid
    add_subtle_grid(ax, axis="both")

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    if title:
        ax.set_title(title, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)

    plt.tight_layout()

    if save_path:
        _save_figure(fig, save_path)

    return fig


def plot_calibration_comparison(
    results: dict[str, tuple[ArrayLike, ArrayLike]],
    num_bins: int = 5,
    title: str | None = None,
    save_path: Path | None = None,
    methods_to_show: list[str] | None = None,
) -> plt.Figure:
    """Plot multiple calibration curves for comparison.

    Clean style with curated colors. Uses fewer bins (5) for smoother curves
    with small sample sizes.

    Args:
        results: Dict mapping method name to (predictions, labels) tuple.
        num_bins: Number of bins for calibration.
        title: Plot title (None for no title - use figure caption).
        save_path: Optional path to save figure.
        methods_to_show: Methods to include (default: pre, post, adversarial).

    Returns:
        Matplotlib figure.
    """
    setup_latex_style()

    # Default to showing only 3 key methods
    if methods_to_show is None:
        methods_to_show = ["exploration_direct", "review_direct", "review_adversarial"]

    # Filter results
    filtered_results = {k: v for k, v in results.items() if k in methods_to_show}
    if not filtered_results:
        filtered_results = results

    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)

    # Perfect calibration line (dashed)
    ax.plot([0, 1], [0, 1], color=COLORS["reference"], linestyle="--", lw=1.5, zorder=1)
    ax.text(0.82, 0.72, "perfect", fontsize=9, color="#999999", rotation=45, va="bottom", style="italic")

    # Use consistent method order
    method_order = ["exploration_direct", "review_direct", "review_adversarial"]
    for method in method_order:
        if method not in filtered_results:
            continue
        preds, labels = filtered_results[method]
        bin_means, bin_accuracies, _ = _compute_calibration_bins(preds, labels, num_bins)
        style = _get_method_style(method)

        # Filter out empty bins
        valid = ~np.isnan(bin_means)
        ax.plot(
            bin_means[valid],
            bin_accuracies[valid],
            lw=LINE_WIDTH,
            color=style["color"],
            marker=style["marker"],
            markersize=style["markersize"] + 1,  # Slightly larger for visibility
            markeredgecolor=style["markeredgecolor"],
            markeredgewidth=style["markeredgewidth"],
            label=_format_method_name(method),
            zorder=3,
        )

    # Add subtle grid
    add_subtle_grid(ax, axis="both")

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    if title:
        ax.set_title(title, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.95)

    plt.tight_layout()

    if save_path:
        _save_figure(fig, save_path)

    return fig


def plot_confidence_histogram(
    predictions: ArrayLike,
    labels: ArrayLike,
    num_bins: int = 20,
    title: str = "Confidence Distribution",
    save_path: Path | None = None,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot histogram of predictions colored by outcome.

    Args:
        predictions: Predicted probabilities in [0, 1].
        labels: Binary labels (0 or 1).
        num_bins: Number of histogram bins.
        title: Plot title.
        save_path: Optional path to save figure.
        ax: Optional axes to plot on.

    Returns:
        Matplotlib figure.
    """
    setup_latex_style()

    predictions = np.asarray(predictions)
    labels = np.asarray(labels)

    if ax is None:
        fig, ax = plt.subplots(figsize=(5.5, 4))
    else:
        fig = ax.get_figure()

    # Separate by outcome
    success_preds = predictions[labels == 1]
    fail_preds = predictions[labels == 0]

    bins = np.linspace(0, 1, num_bins + 1)

    # Use curated colors
    ax.hist(
        success_preds,
        bins=bins,
        alpha=0.75,
        label=f"Success (n={len(success_preds)})",
        color=COLORS["success"],
        edgecolor="white",
        linewidth=0.5,
        zorder=2,
    )
    ax.hist(
        fail_preds,
        bins=bins,
        alpha=0.75,
        label=f"Failure (n={len(fail_preds)})",
        color=COLORS["failure"],
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
    )

    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Count")
    ax.set_title(_format_method_name(title) if "_" in title else title)
    ax.legend()

    if save_path:
        _save_figure(fig, save_path)

    return fig


def plot_confidence_histograms_comparison(
    results: dict[str, tuple[ArrayLike, ArrayLike]],
    num_bins: int = 15,
    save_path: Path | None = None,
    methods_to_show: list[str] | None = None,
) -> plt.Figure:
    """Plot confidence histograms for multiple methods.

    Clean horizontal layout (1x3 for 3 methods), curated colors.

    Args:
        results: Dict mapping method name to (predictions, labels) tuple.
        num_bins: Number of histogram bins.
        save_path: Optional path to save figure.
        methods_to_show: Methods to include (default: pre, post, adversarial).

    Returns:
        Matplotlib figure.
    """
    setup_latex_style()

    # Default to showing only 3 key methods
    if methods_to_show is None:
        methods_to_show = ["exploration_direct", "review_direct", "review_adversarial"]

    # Filter and order results
    ordered_keys = [k for k in methods_to_show if k in results]
    if not ordered_keys:
        ordered_keys = list(results.keys())[:3]  # Fallback to first 3

    n_methods = len(ordered_keys)

    # Horizontal layout for cleaner appearance
    fig, axes = plt.subplots(1, n_methods, figsize=(4.0 * n_methods, 3.5))
    if n_methods == 1:
        axes = [axes]

    bins = np.linspace(0, 1, num_bins + 1)

    for idx, method in enumerate(ordered_keys):
        preds, labels = results[method]
        preds = np.asarray(preds)
        labels = np.asarray(labels)
        ax = axes[idx]

        # Separate by outcome
        success_preds = preds[labels == 1]
        fail_preds = preds[labels == 0]

        ax.hist(
            success_preds,
            bins=bins,
            alpha=0.75,
            label=f"Success (n={len(success_preds)})",
            color=COLORS["success"],
            edgecolor="white",
            linewidth=0.5,
            zorder=2,
        )
        ax.hist(
            fail_preds,
            bins=bins,
            alpha=0.75,
            label=f"Failure (n={len(fail_preds)})",
            color=COLORS["failure"],
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )

        title = _format_method_name(method)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted Probability", fontsize=10)
        if idx == 0:
            ax.set_ylabel("Count", fontsize=10)
        ax.tick_params(labelsize=9)
        ax.legend(fontsize=8, loc="upper left", framealpha=0.95, edgecolor="none")
        ax.set_xlim(0, 1)

        # Add subtle grid
        add_subtle_grid(ax, axis="y")

    plt.tight_layout()

    if save_path:
        _save_figure(fig, save_path)

    return fig


def plot_auroc_progression(
    model_data: dict[str, dict[str, Any]],
    title: str | None = None,
    save_path: Path | None = None,
    show_logos: bool = True,
) -> plt.Figure:
    """Plot AUROC vs checkpoint % - clean line plot without error bars.

    Shows how discrimination ability evolves as trajectory progresses.

    Args:
        model_data: Dict mapping model name to {
            'checkpoints': list of checkpoint percentages,
            'aurocs': list of AUROC values,
        }
        title: Plot title (None for no title).
        save_path: Optional path to save figure.
        show_logos: If True, show model logos at end of lines.

    Returns:
        Matplotlib figure.
    """
    setup_latex_style()

    fig, ax = plt.subplots(figsize=(5.0, 3.8))

    # Load logos for legend
    logos = {}
    if show_logos:
        for model_name in model_data.keys():
            model_lower = model_name.lower()
            if "gpt" in model_lower:
                logo_key = "gpt"
            elif "gemini" in model_lower:
                logo_key = "gemini"
            elif "claude" in model_lower:
                logo_key = "claude"
            else:
                logo_key = None
            if logo_key:
                logo = _load_logo(logo_key, size=80)
                if logo is not None:
                    logos[model_name] = logo

    for model_name, data in model_data.items():
        checkpoints = data["checkpoints"]
        aurocs = data["aurocs"]

        style = _get_model_style(model_name)

        ax.plot(
            checkpoints,
            aurocs,
            marker=style["marker"],
            markersize=style["markersize"] + 2,
            markeredgecolor=style["markeredgecolor"],
            markeredgewidth=style["markeredgewidth"],
            linewidth=LINE_WIDTH,
            color=style["color"],
            label=_format_model_name(model_name),
            zorder=3,
        )

    # Shade below-chance region (like overconfident region in calibration figure)
    ax.fill_between([15, 90], 0.3, 0.5, alpha=0.08, color="#888888", zorder=0)
    ax.axhline(y=0.5, color=COLORS["reference"], linestyle="-", linewidth=1, zorder=1)
    ax.text(23, 0.505, "chance", fontsize=FONTSIZE["reference_text"], color="#888888",
            va="bottom", style="italic")

    # Add subtle grid
    add_subtle_grid(ax, axis="y")

    ax.set_xlabel("Trajectory Progress", fontsize=FONTSIZE["axis_label"])
    ax.set_ylabel("AUROC", fontsize=FONTSIZE["axis_label"])
    ax.tick_params(labelsize=FONTSIZE["tick_label"])
    if title:
        ax.set_title(title, fontweight="bold", fontsize=FONTSIZE["title"])
    ax.set_xlim(20, 80)
    ax.set_ylim(0.42, 0.72)
    ax.set_xticks([25, 50, 75])
    ax.set_xticklabels(["25%", "50%", "75%"])

    # Clean legend with logos (matching calibration figure style)
    if show_logos and logos:
        # Place logo + colored model name in upper-left area
        legend_x = 0.03
        legend_y_start = 0.97
        v_spacing = 0.12

        # Get model order for consistent layout
        model_order_map = {"gpt": 0, "gemini": 1, "claude": 2}
        sorted_models = sorted(
            model_data.keys(),
            key=lambda m: model_order_map.get(m.lower().split("-")[0], 99)
        )

        for i, model_name in enumerate(sorted_models):
            y_pos = legend_y_start - i * v_spacing
            style = _get_model_style(model_name)

            if model_name in logos:
                imagebox = OffsetImage(logos[model_name], zoom=0.12)
                imagebox.image.axes = ax
                ab = AnnotationBbox(
                    imagebox,
                    (legend_x + 0.03, y_pos),
                    xycoords="axes fraction",
                    frameon=False,
                    zorder=10,
                )
                ax.add_artist(ab)

            ax.text(
                legend_x + 0.08,
                y_pos,
                _format_model_name(model_name),
                transform=ax.transAxes,
                ha="left",
                va="center",
                fontsize=FONTSIZE["legend"],
                color=style["color"],
                fontweight="bold",
            )
    else:
        ax.legend(loc="upper left", framealpha=0.95, edgecolor="none", fontsize=FONTSIZE["legend"])

    plt.tight_layout()

    if save_path:
        _save_figure(fig, save_path)

    return fig


def plot_confidence_trajectories(
    checkpoint_data: dict[int, dict[str, Any]],
    title: str | None = None,
    save_path: Path | None = None,
    max_instances: int = 100,
) -> plt.Figure:
    """Plot mean confidence trajectories for success vs failure.

    Clean plot showing how predictions evolve, with gap annotation.

    Args:
        checkpoint_data: Dict mapping checkpoint % to {
            'predictions': np.ndarray,
            'labels': np.ndarray,
            'instance_ids': list[str],
        }
        title: Plot title (None for no title).
        save_path: Optional path to save figure.
        max_instances: Maximum instances to plot (for clarity).

    Returns:
        Matplotlib figure.
    """
    setup_latex_style()

    fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)

    # Find common instances across all checkpoints
    all_ids = None
    for data in checkpoint_data.values():
        ids = set(data["instance_ids"])
        if all_ids is None:
            all_ids = ids
        else:
            all_ids &= ids

    if not all_ids:
        ax.text(0.5, 0.5, "No common instances", ha="center", va="center")
        return fig

    common_ids = sorted(all_ids)[:max_instances]

    # Build index maps for each checkpoint
    id_to_idx = {}
    for checkpoint, data in checkpoint_data.items():
        id_to_idx[checkpoint] = {iid: i for i, iid in enumerate(data["instance_ids"])}

    # Get labels from first checkpoint (should be same across all)
    first_checkpoint = min(checkpoint_data.keys())
    labels = {
        iid: checkpoint_data[first_checkpoint]["labels"][id_to_idx[first_checkpoint][iid]]
        for iid in common_ids
    }

    checkpoints = sorted(checkpoint_data.keys())

    # Compute means only (no std bands - they overlap and look bad)
    success_means = []
    failure_means = []

    for cp in checkpoints:
        cp_preds = checkpoint_data[cp]["predictions"]
        cp_idx = id_to_idx[cp]

        success_preds = [cp_preds[cp_idx[iid]] for iid in common_ids if labels[iid] == 1]
        failure_preds = [cp_preds[cp_idx[iid]] for iid in common_ids if labels[iid] == 0]

        success_means.append(np.mean(success_preds) if success_preds else np.nan)
        failure_means.append(np.mean(failure_preds) if failure_preds else np.nan)

    success_means = np.array(success_means)
    failure_means = np.array(failure_means)

    n_success = sum(1 for v in labels.values() if v == 1)
    n_failure = sum(1 for v in labels.values() if v == 0)

    # Plot mean lines with curated colors
    ax.plot(
        checkpoints,
        success_means,
        color=COLORS["success"],
        linewidth=LINE_WIDTH,
        marker="o",
        markersize=9,
        markeredgecolor="black",
        markeredgewidth=0.8,
        label=f"Success (n={n_success})",
        zorder=3,
    )
    ax.plot(
        checkpoints,
        failure_means,
        color=COLORS["failure"],
        linewidth=LINE_WIDTH,
        marker="s",
        markersize=9,
        markeredgecolor="black",
        markeredgewidth=0.8,
        label=f"Failure (n={n_failure})",
        zorder=3,
    )

    # Base rate reference with label
    base_rate = n_success / (n_success + n_failure)
    ax.axhline(y=base_rate, color=COLORS["reference"], linestyle="-", linewidth=1, zorder=1)
    ax.text(77, base_rate + 0.02, f"base rate ({base_rate:.0%})", fontsize=8, color="#888888", va="bottom")

    # Add subtle grid
    add_subtle_grid(ax, axis="y")

    ax.set_xlabel("Trajectory Progress")
    ax.set_ylabel("Mean Confidence")
    if title:
        ax.set_title(title, fontweight="bold")
    ax.set_xlim(20, 80)
    ax.set_ylim(0.3, 0.8)
    ax.set_xticks(checkpoints)
    ax.set_xticklabels([f"{cp}%" for cp in checkpoints])
    ax.legend(loc="upper right", framealpha=0.95, edgecolor="none", fontsize=9)

    plt.tight_layout()

    if save_path:
        _save_figure(fig, save_path)

    return fig


def plot_confidence_trajectory_lines(
    instance_data: dict[str, dict[str, dict[str, Any]]],
    save_path: Path | None = None,
    max_instances: int | None = None,
    alpha_individual: float = 0.08,
    show_means: bool = True,
    show_bands: bool = True,
    show_logos: bool = True,
    show_individual: bool = True,
    y_min: float = 0.0,
) -> plt.Figure:
    """Plot individual confidence trajectories with mean overlay and CI bands.

    Creates a 1x3 panel layout (one per model: GPT, Gemini, Claude) showing
    individual trajectory lines colored by outcome, with mean lines and
    shaded confidence bands.

    Args:
        instance_data: model -> instance_id -> {
            'checkpoints': [25, 50, 75],
            'confidences': [0.6, 0.7, 0.8],
            'label': True/False
        }
        save_path: Optional path to save figure.
        max_instances: Maximum instances to plot per model (None for all).
        alpha_individual: Alpha for individual trajectory lines.
        show_means: If True, overlay thick mean lines with markers.
        show_bands: If True, show shaded IQR bands around means.
        show_logos: If True, show model logos above panel titles.
        show_individual: If True, show individual trajectory lines.
        y_min: Minimum y-axis value (default 0.0).

    Returns:
        Matplotlib figure.
    """
    setup_latex_style()

    # Model order and display names
    model_order = {"gpt": 0, "gemini": 1, "claude": 2}
    models = sorted(
        instance_data.keys(),
        key=lambda m: model_order.get(m.lower().split("-")[0], 99)
    )

    n_models = len(models)
    if n_models == 0:
        fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig

    fig, axes = plt.subplots(1, n_models, figsize=(4.2 * n_models, 4.0), sharey=True)
    if n_models == 1:
        axes = [axes]

    # Load logos
    logos = {}
    if show_logos:
        for model in models:
            model_lower = model.lower()
            if "gpt" in model_lower:
                logo_key = "gpt"
            elif "gemini" in model_lower:
                logo_key = "gemini"
            elif "claude" in model_lower:
                logo_key = "claude"
            else:
                logo_key = None
            if logo_key:
                logo = _load_logo(logo_key, size=120)
                if logo is not None:
                    logos[model] = logo

    checkpoints = [25, 50, 75]

    for ax, model in zip(axes, models):
        trajectories = instance_data[model]

        # Separate by outcome
        success_trajs = []
        failure_trajs = []
        for instance_id, traj in trajectories.items():
            if traj["label"]:
                success_trajs.append(traj)
            else:
                failure_trajs.append(traj)

        # Limit instances if requested
        if max_instances is not None:
            success_trajs = success_trajs[:max_instances]
            failure_trajs = failure_trajs[:max_instances]

        # Plot individual trajectories with very low opacity (background texture)
        if show_individual:
            # Plot failure trajectories first (behind)
            for traj in failure_trajs:
                ax.plot(
                    traj["checkpoints"],
                    traj["confidences"],
                    color=COLORS["failure"],
                    alpha=alpha_individual,
                    linewidth=0.6,
                    zorder=1,
                )

            # Plot success trajectories
            for traj in success_trajs:
                ax.plot(
                    traj["checkpoints"],
                    traj["confidences"],
                    color=COLORS["success"],
                    alpha=alpha_individual,
                    linewidth=0.6,
                    zorder=2,
                )

        # Compute statistics for bands and means
        def compute_stats(trajs):
            means, stds, q25s, q75s = [], [], [], []
            for cp in checkpoints:
                cp_values = [t["confidences"][t["checkpoints"].index(cp)]
                            for t in trajs if cp in t["checkpoints"]]
                if cp_values:
                    means.append(np.mean(cp_values))
                    stds.append(np.std(cp_values) / np.sqrt(len(cp_values)))  # SEM
                    q25s.append(np.percentile(cp_values, 25))
                    q75s.append(np.percentile(cp_values, 75))
                else:
                    means.append(np.nan)
                    stds.append(np.nan)
                    q25s.append(np.nan)
                    q75s.append(np.nan)
            return means, stds, q25s, q75s

        # Plot bands and means
        if show_means and (success_trajs or failure_trajs):
            # Failure: use dashed line with square markers
            if failure_trajs:
                f_means, f_stds, f_q25, f_q75 = compute_stats(failure_trajs)
                f_means = np.array(f_means)
                f_stds = np.array(f_stds)

                if show_bands:
                    # Use SEM-based bands (tighter, more interpretable)
                    ax.fill_between(
                        checkpoints,
                        f_means - 1.96 * f_stds,
                        f_means + 1.96 * f_stds,
                        color=COLORS["failure"],
                        alpha=0.15,
                        zorder=3,
                        linewidth=0,
                    )

                ax.plot(
                    checkpoints,
                    f_means,
                    color=COLORS["failure"],
                    linewidth=3.5,
                    linestyle="--",
                    marker="s",
                    markersize=12,
                    markeredgecolor="white",
                    markeredgewidth=2.5,
                    markerfacecolor=COLORS["failure"],
                    label=f"Failure (n={len(failure_trajs)})",
                    zorder=6,
                )

            # Success: use solid line with circle markers
            if success_trajs:
                s_means, s_stds, s_q25, s_q75 = compute_stats(success_trajs)
                s_means = np.array(s_means)
                s_stds = np.array(s_stds)

                if show_bands:
                    ax.fill_between(
                        checkpoints,
                        s_means - 1.96 * s_stds,
                        s_means + 1.96 * s_stds,
                        color=COLORS["success"],
                        alpha=0.15,
                        zorder=4,
                        linewidth=0,
                    )

                ax.plot(
                    checkpoints,
                    s_means,
                    color=COLORS["success"],
                    linewidth=3.5,
                    linestyle="-",
                    marker="o",
                    markersize=12,
                    markeredgecolor="white",
                    markeredgewidth=2.5,
                    markerfacecolor=COLORS["success"],
                    label=f"Success (n={len(success_trajs)})",
                    zorder=7,
                )

        # Get model color for title
        model_lower = model.lower()
        if "gpt" in model_lower:
            title_color = COLORS["gpt"]
        elif "gemini" in model_lower:
            title_color = COLORS["gemini"]
        elif "claude" in model_lower:
            title_color = COLORS["claude"]
        else:
            title_color = "black"

        # Add logo and title
        if show_logos and model in logos:
            imagebox = OffsetImage(logos[model], zoom=LOGO_ZOOM)
            imagebox.image.axes = ax
            ab = AnnotationBbox(
                imagebox,
                (0.12, 1.08),
                xycoords="axes fraction",
                frameon=False,
                zorder=10,
            )
            ax.add_artist(ab)
            ax.set_title(
                _format_model_name(model),
                fontsize=FONTSIZE["title_with_logo"],
                fontweight="bold",
                color=title_color,
                loc="center",
                pad=10,
            )
        else:
            ax.set_title(
                _format_model_name(model),
                fontsize=FONTSIZE["title"],
                fontweight="bold",
                color=title_color,
            )

        ax.set_xlabel("Trajectory Progress", fontsize=FONTSIZE["axis_label"])
        if ax == axes[0]:
            ax.set_ylabel("Confidence", fontsize=FONTSIZE["axis_label"])
        ax.tick_params(labelsize=FONTSIZE["tick_label"])
        ax.set_xlim(20, 80)
        ax.set_ylim(y_min, 1.02)
        ax.set_xticks(checkpoints)
        ax.set_xticklabels(["25%", "50%", "75%"])

        # Legend in lower left (less competition with data at top)
        if show_means:
            ax.legend(fontsize=FONTSIZE["legend"], loc="lower left",
                     framealpha=0.95, edgecolor="none")

        add_subtle_grid(ax, axis="y")

    plt.tight_layout()
    if show_logos and logos:
        plt.subplots_adjust(top=0.90)

    if save_path:
        _save_figure(fig, save_path)

    return fig


def plot_delta_confidence_analysis(
    instance_data: dict[str, dict[str, dict[str, Any]]],
    save_path: Path | None = None,
    show_points: bool = True,
    show_logos: bool = True,
) -> tuple[plt.Figure, dict[str, dict[str, float]]]:
    """Analyze correlation between confidence change and outcome.

    Creates a 1x3 panel figure (one per model) with split violin plots showing
    the distribution of Δconf (conf_75% - conf_25%) by outcome. Matches the
    calibration figure aesthetic with model logos and colored titles.

    Args:
        instance_data: model -> instance_id -> {
            'checkpoints': [25, 50, 75],
            'confidences': [0.6, 0.7, 0.8],
            'label': True/False
        }
        save_path: Optional path to save figure.
        show_points: If True, overlay individual data points.
        show_logos: If True, show model logos above panel titles.

    Returns:
        Tuple of (figure, stats_dict) where stats_dict contains:
        model -> {
            'success_mean_delta': float,
            'failure_mean_delta': float,
            'point_biserial_r': float,
            'p_value': float,
        }
    """
    from matplotlib.patches import Patch
    from scipy import stats as scipy_stats

    setup_latex_style()

    # Model order and display names
    model_order = {"gpt": 0, "gemini": 1, "claude": 2}
    models = sorted(
        instance_data.keys(),
        key=lambda m: model_order.get(m.lower().split("-")[0], 99)
    )

    if not models:
        fig, ax = plt.subplots(figsize=FIGSIZE_SINGLE)
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        return fig, {}

    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(4.2 * n_models, 4.0), sharey=True)
    if n_models == 1:
        axes = [axes]

    # Load logos
    logos = {}
    if show_logos:
        for model in models:
            model_lower = model.lower()
            if "gpt" in model_lower:
                logo_key = "gpt"
            elif "gemini" in model_lower:
                logo_key = "gemini"
            elif "claude" in model_lower:
                logo_key = "claude"
            else:
                logo_key = None
            if logo_key:
                logo = _load_logo(logo_key, size=120)
                if logo is not None:
                    logos[model] = logo

    all_stats = {}
    width = 0.45

    for panel_idx, (ax, model) in enumerate(zip(axes, models)):
        trajectories = instance_data[model]

        # Compute delta confidence for each instance
        success_deltas = []
        failure_deltas = []

        for instance_id, traj in trajectories.items():
            checkpoints = traj["checkpoints"]
            confidences = traj["confidences"]

            if 25 in checkpoints and 75 in checkpoints:
                idx_25 = checkpoints.index(25)
                idx_75 = checkpoints.index(75)
                delta = confidences[idx_75] - confidences[idx_25]

                if traj["label"]:
                    success_deltas.append(delta)
                else:
                    failure_deltas.append(delta)

        violin_success = success_deltas if success_deltas else [0]
        violin_failure = failure_deltas if failure_deltas else [0]

        # Compute statistics
        all_deltas = success_deltas + failure_deltas
        all_labels = [1] * len(success_deltas) + [0] * len(failure_deltas)

        if len(all_deltas) > 2 and len(set(all_labels)) > 1:
            r, p_value = scipy_stats.pointbiserialr(all_labels, all_deltas)
        else:
            r, p_value = np.nan, np.nan

        all_stats[model] = {
            "success_mean_delta": float(np.mean(success_deltas)) if success_deltas else np.nan,
            "failure_mean_delta": float(np.mean(failure_deltas)) if failure_deltas else np.nan,
            "success_n": len(success_deltas),
            "failure_n": len(failure_deltas),
            "point_biserial_r": float(r),
            "p_value": float(p_value),
        }

        # Draw split violin - success left, failure right
        pos = 0  # Single position per panel

        # Success violin (clipped to left)
        if len(violin_success) > 1:
            parts_s = ax.violinplot(
                violin_success,
                positions=[pos],
                widths=width,
                showmeans=False,
                showextrema=False,
            )
            for pc in parts_s["bodies"]:
                m = np.mean(pc.get_paths()[0].vertices[:, 0])
                pc.get_paths()[0].vertices[:, 0] = np.clip(
                    pc.get_paths()[0].vertices[:, 0], -np.inf, m
                )
                pc.set_facecolor(COLORS["success"])
                pc.set_edgecolor("white")
                pc.set_linewidth(1.2)
                pc.set_alpha(0.75)

            # Median line and mean marker
            median_s = np.median(violin_success)
            mean_s = np.mean(violin_success)
            ax.hlines(median_s, pos - width/2.5, pos - 0.02,
                     color="white", linewidth=2.5, zorder=5)
            ax.hlines(median_s, pos - width/2.5, pos - 0.02,
                     color=COLORS["success"], linewidth=1.5, zorder=6)
            ax.scatter([pos - width/4], [mean_s], color="white",
                      s=90, marker="o", edgecolor="white", linewidth=0, zorder=5)
            ax.scatter([pos - width/4], [mean_s], color=COLORS["success"],
                      s=60, marker="o", edgecolor="white", linewidth=1.5, zorder=6)

        # Failure violin (clipped to right)
        if len(violin_failure) > 1:
            parts_f = ax.violinplot(
                violin_failure,
                positions=[pos],
                widths=width,
                showmeans=False,
                showextrema=False,
            )
            for pc in parts_f["bodies"]:
                m = np.mean(pc.get_paths()[0].vertices[:, 0])
                pc.get_paths()[0].vertices[:, 0] = np.clip(
                    pc.get_paths()[0].vertices[:, 0], m, np.inf
                )
                pc.set_facecolor(COLORS["failure"])
                pc.set_edgecolor("white")
                pc.set_linewidth(1.2)
                pc.set_alpha(0.75)

            median_f = np.median(violin_failure)
            mean_f = np.mean(violin_failure)
            ax.hlines(median_f, pos + 0.02, pos + width/2.5,
                     color="white", linewidth=2.5, zorder=5)
            ax.hlines(median_f, pos + 0.02, pos + width/2.5,
                     color=COLORS["failure"], linewidth=1.5, zorder=6)
            ax.scatter([pos + width/4], [mean_f], color="white",
                      s=90, marker="s", edgecolor="white", linewidth=0, zorder=5)
            ax.scatter([pos + width/4], [mean_f], color=COLORS["failure"],
                      s=60, marker="s", edgecolor="white", linewidth=1.5, zorder=6)

        # Individual data points with jitter
        if show_points:
            np.random.seed(42 + panel_idx)
            jitter_s = np.random.uniform(-width/3, -0.03, len(violin_success))
            ax.scatter(
                pos + jitter_s,
                violin_success,
                color=COLORS["success"],
                alpha=0.4,
                s=10,
                edgecolor="none",
                zorder=3,
            )
            jitter_f = np.random.uniform(0.03, width/3, len(violin_failure))
            ax.scatter(
                pos + jitter_f,
                violin_failure,
                color=COLORS["failure"],
                alpha=0.4,
                s=10,
                edgecolor="none",
                zorder=3,
            )

        # Reference line at zero with annotation
        ax.axhline(y=0, color=COLORS["reference"], linestyle="--", linewidth=1.5, zorder=1)
        ax.text(0.97, 0.52, "no change", fontsize=FONTSIZE["reference_text"],
                color="#888888", va="bottom", ha="right", style="italic",
                transform=ax.transAxes)

        # Get model color for title
        model_lower = model.lower()
        if "gpt" in model_lower:
            title_color = COLORS["gpt"]
        elif "gemini" in model_lower:
            title_color = COLORS["gemini"]
        elif "claude" in model_lower:
            title_color = COLORS["claude"]
        else:
            title_color = "black"

        # Add logo and title (matching calibration figure aesthetic)
        if show_logos and model in logos:
            imagebox = OffsetImage(logos[model], zoom=LOGO_ZOOM)
            imagebox.image.axes = ax
            ab = AnnotationBbox(
                imagebox,
                (0.12, 1.08),
                xycoords="axes fraction",
                frameon=False,
                zorder=10,
            )
            ax.add_artist(ab)
            ax.set_title(
                _format_model_name(model),
                fontsize=FONTSIZE["title_with_logo"],
                fontweight="bold",
                color=title_color,
                loc="center",
                pad=10,
            )
        else:
            ax.set_title(
                _format_model_name(model),
                fontsize=FONTSIZE["title"],
                fontweight="bold",
                color=title_color,
            )

        if ax == axes[0]:
            ax.set_ylabel(r"$\Delta$confidence (75% $-$ 25%)", fontsize=FONTSIZE["axis_label"])
        ax.tick_params(labelsize=FONTSIZE["tick_label"])
        ax.set_xlim(-0.5, 0.5)
        ax.set_xticks([])

        # Legend (matching calibration figure style)
        legend_elements = [
            Patch(facecolor=COLORS["success"], alpha=0.75, edgecolor="white",
                  linewidth=1, label="Success"),
            Patch(facecolor=COLORS["failure"], alpha=0.75, edgecolor="white",
                  linewidth=1, label="Failure"),
        ]
        ax.legend(handles=legend_elements, loc="upper right",
                 fontsize=FONTSIZE["legend"], framealpha=0.95, edgecolor="none")

        add_subtle_grid(ax, axis="y")

    plt.tight_layout()
    if show_logos and logos:
        plt.subplots_adjust(top=0.90)

    if save_path:
        _save_figure(fig, save_path)

    return fig, all_stats


def plot_hero_results(
    data: dict[str, dict[str, dict[str, float]]],
    save_path: Path | None = None,
    vertical: bool = True,
) -> plt.Figure:
    """Create 3-panel bar chart: AUROC, ECE, Mean Confidence.

    Hero figure for the paper showing key results across models and methods.

    Args:
        data: model -> method -> {"auroc", "ece", "mean_conf", "base_rate"}
              Models: "gpt", "gemini", "claude"
              Methods: "pre", "post", "adv"
        save_path: Optional path to save figure.
        vertical: If True, stack panels vertically (fits single column).
                  If False, arrange horizontally (wide format).

    Returns:
        Matplotlib figure.
    """
    setup_latex_style()

    # Panel configuration
    panels = [
        ("auroc", "AUROC", True, 0.5),   # (key, label, higher_is_better, reference_value)
        ("ece", "ECE", False, None),
        ("mean_conf", "Mean Confidence", None, None),  # Reference lines vary by model
    ]

    # Model and method configuration
    models = ["gpt", "gemini", "claude"]
    methods = ["pre", "post", "adv"]

    model_colors = {
        "gpt": COLORS["gpt"],
        "gemini": COLORS["gemini"],
        "claude": COLORS["claude"],
    }

    model_display = {
        "gpt": "GPT-5.2-Codex",
        "gemini": "Gemini-3-Pro",
        "claude": "Claude Opus 4.5",
    }

    method_display = {
        "pre": "Pre-Exec",
        "post": "Post-Exec",
        "adv": "Adv. Post",
    }

    # Create figure - vertical (3x1) or horizontal (1x3) layout
    if vertical:
        fig, axes = plt.subplots(3, 1, figsize=(3.3, 7.5))
    else:
        fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.5))

    n_models = len(models)
    n_methods = len(methods)
    bar_width = 0.25
    x = np.arange(n_methods)

    for panel_idx, (key, ylabel, higher_better, ref_value) in enumerate(panels):
        ax = axes[panel_idx]

        for model_idx, model in enumerate(models):
            if model not in data:
                continue

            values = []
            for method in methods:
                if method in data[model]:
                    values.append(data[model][method].get(key, 0))
                else:
                    values.append(0)

            positions = x + (model_idx - 1) * bar_width

            bars = ax.bar(
                positions,
                values,
                bar_width * 0.85,
                label=model_display[model],
                color=model_colors[model],
                edgecolor="black",
                linewidth=0.5,
                zorder=3,
            )

            # Add value labels on top of bars
            for bar, val in zip(bars, values):
                if val > 0:
                    ax.annotate(
                        f"{val:.2f}",
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 2),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        color="#444444",
                    )

        # Add reference line for AUROC (chance level)
        if key == "auroc" and ref_value is not None:
            ax.axhline(y=ref_value, color=COLORS["reference"], linestyle="-", lw=1.5, zorder=1)
            ax.text(
                -0.45, ref_value,
                "chance",
                fontsize=7,
                color="#888888",
                va="center",
                ha="right",
            )

        # Add base rate reference lines for mean confidence panel
        if key == "mean_conf":
            base_rates = {
                "gpt": data.get("gpt", {}).get("pre", {}).get("base_rate", 0.344),
                "gemini": data.get("gemini", {}).get("pre", {}).get("base_rate", 0.215),
                "claude": data.get("claude", {}).get("pre", {}).get("base_rate", 0.270),
            }
            # Add a single annotation explaining base rates
            avg_base = np.mean(list(base_rates.values()))
            ax.axhline(y=avg_base, color=COLORS["reference"], linestyle="--", lw=1, zorder=1)
            ax.text(
                2.6, avg_base,
                "avg base rate",
                fontsize=6,
                color="#888888",
                va="center",
                ha="left",
            )

        # Styling
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels([method_display[m] for m in methods], fontsize=9)
        ax.tick_params(labelsize=9)

        # Add subtle grid
        add_subtle_grid(ax, axis="y")

        # Set y-axis limits based on metric
        if key == "auroc":
            ax.set_ylim(0.4, 0.8)
        elif key == "ece":
            ax.set_ylim(0, 0.55)
        else:
            ax.set_ylim(0, 0.85)

        # Only add legend to first panel
        if panel_idx == 0:
            ax.legend(
                loc="upper right" if vertical else "upper left",
                fontsize=7,
                framealpha=0.95,
                edgecolor="none",
            )

        # Panel label
        ax.text(
            -0.15 if vertical else -0.12,
            1.02,
            f"({chr(97 + panel_idx)})",
            transform=ax.transAxes,
            fontsize=10,
            fontweight="bold",
            va="bottom",
        )

    plt.tight_layout()

    if save_path:
        _save_figure(fig, save_path)

    return fig


def plot_confidence_histograms_by_model(
    model_results: dict[str, dict[str, tuple[ArrayLike, ArrayLike]]],
    num_bins: int = 15,
    save_path: Path | None = None,
    method_to_show: str = "review_direct",
    show_base_rate: bool = True,
    show_logos: bool = True,
    style: str = "mirrored",
) -> plt.Figure:
    """Plot confidence histograms for each model, colored by outcome.

    Creates a 1x3 panel figure showing post-execution confidence distributions
    for GPT, Gemini, and Claude. Success cases in green, failures in red.

    Args:
        model_results: model -> method -> (predictions, labels)
        num_bins: Number of histogram bins.
        save_path: Optional path to save figure.
        method_to_show: Which method to display (default: "review_direct").
        show_base_rate: If True, show vertical dashed line for base rate.
        show_logos: If True, show model logos above panel titles.
        style: "mirrored" for back-to-back, "step" for outlines, "bars" for side-by-side.

    Returns:
        Matplotlib figure.
    """
    setup_latex_style()

    # Model order and display names
    model_order = {"gpt": 0, "gemini": 1, "claude": 2}
    models = sorted(
        model_results.keys(),
        key=lambda m: model_order.get(m.lower().split("-")[0], 99)
    )

    n_models = len(models)
    # Extra height for logo space
    fig, axes = plt.subplots(1, n_models, figsize=(4.2 * n_models, 4.0), sharey=True)
    if n_models == 1:
        axes = [axes]

    bins = np.linspace(0, 1, num_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_width = bins[1] - bins[0]

    # Load logos
    logos = {}
    if show_logos:
        for model in models:
            model_lower = model.lower()
            if "gpt" in model_lower:
                logo_key = "gpt"
            elif "gemini" in model_lower:
                logo_key = "gemini"
            elif "claude" in model_lower:
                logo_key = "claude"
            else:
                logo_key = None
            if logo_key:
                logo = _load_logo(logo_key, size=120)
                if logo is not None:
                    logos[model] = logo

    for ax, model in zip(axes, models):
        if model not in model_results:
            continue

        method_data = model_results[model]

        # Find the method to show
        if method_to_show in method_data:
            preds, labels = method_data[method_to_show]
        else:
            # Fallback to first available method
            first_method = next(iter(method_data.keys()))
            preds, labels = method_data[first_method]

        preds = np.asarray(preds)
        labels = np.asarray(labels)

        # Separate by outcome
        success_preds = preds[labels == 1]
        fail_preds = preds[labels == 0]

        # Compute histogram counts
        success_counts, _ = np.histogram(success_preds, bins=bins)
        fail_counts, _ = np.histogram(fail_preds, bins=bins)

        if style == "mirrored":
            # Mirrored/back-to-back histogram - success up, failure down
            # This dramatically shows overlap as mirror symmetry
            ax.bar(
                bin_centers,
                success_counts,
                width=bin_width * 0.9,
                label=f"Success (n={len(success_preds)})",
                color=COLORS["success"],
                edgecolor="white",
                linewidth=0.5,
                zorder=2,
            )
            ax.bar(
                bin_centers,
                -fail_counts,  # Negative to go downward
                width=bin_width * 0.9,
                label=f"Failure (n={len(fail_preds)})",
                color=COLORS["failure"],
                edgecolor="white",
                linewidth=0.5,
                zorder=2,
            )
            # Add horizontal line at y=0
            ax.axhline(y=0, color="black", linewidth=1.0, zorder=3)

        elif style == "step":
            # Step histogram - unfilled outlines, no occlusion
            ax.hist(
                success_preds,
                bins=bins,
                histtype="step",
                linewidth=2.5,
                label=f"Success (n={len(success_preds)})",
                color=COLORS["success"],
                zorder=3,
            )
            ax.hist(
                fail_preds,
                bins=bins,
                histtype="step",
                linewidth=2.5,
                linestyle="--",
                label=f"Failure (n={len(fail_preds)})",
                color=COLORS["failure"],
                zorder=2,
            )
            # Add light fill for visual weight
            ax.hist(
                success_preds,
                bins=bins,
                histtype="stepfilled",
                alpha=0.25,
                color=COLORS["success"],
                zorder=1,
            )
            ax.hist(
                fail_preds,
                bins=bins,
                histtype="stepfilled",
                alpha=0.25,
                color=COLORS["failure"],
                zorder=0,
            )
        else:
            # Side-by-side bars with hatching
            bar_width_adj = bin_width * 0.38
            gap = bin_width * 0.02

            ax.bar(
                bin_centers - bar_width_adj / 2 - gap / 2,
                success_counts,
                width=bar_width_adj,
                label=f"Success (n={len(success_preds)})",
                color=COLORS["success"],
                edgecolor="#1a7a1a",
                linewidth=0.8,
                zorder=2,
            )
            ax.bar(
                bin_centers + bar_width_adj / 2 + gap / 2,
                fail_counts,
                width=bar_width_adj,
                label=f"Failure (n={len(fail_preds)})",
                color=COLORS["failure"],
                edgecolor="#a31f1f",
                linewidth=0.8,
                hatch="//",
                zorder=3,
            )

        # Add base rate line
        if show_base_rate:
            base_rate = np.mean(labels)
            ax.axvline(
                x=base_rate,
                color=COLORS["reference"],
                linestyle="--",
                linewidth=1.5,
                zorder=4,
                label=f"Base rate ({base_rate:.0%})",
            )

        # Get model color for title
        model_lower = model.lower()
        if "gpt" in model_lower:
            title_color = COLORS["gpt"]
        elif "gemini" in model_lower:
            title_color = COLORS["gemini"]
        elif "claude" in model_lower:
            title_color = COLORS["claude"]
        else:
            title_color = "black"

        # Add logo to the left of title
        if show_logos and model in logos:
            imagebox = OffsetImage(logos[model], zoom=LOGO_ZOOM)
            imagebox.image.axes = ax
            ab = AnnotationBbox(
                imagebox,
                (0.12, 1.08),
                xycoords="axes fraction",
                frameon=False,
                zorder=10,
            )
            ax.add_artist(ab)
            # Title to the right of logo
            ax.set_title(
                _format_model_name(model),
                fontsize=FONTSIZE["title_with_logo"],
                fontweight="bold",
                color=title_color,
                loc="center",
                pad=10,
            )
        else:
            ax.set_title(
                _format_model_name(model),
                fontsize=FONTSIZE["title"],
                fontweight="bold",
                color=title_color,
            )

        ax.set_xlabel("Predicted Probability", fontsize=FONTSIZE["axis_label"])
        ax.tick_params(labelsize=FONTSIZE["tick_label"])
        ax.set_xlim(0, 1)

        if style == "mirrored":
            # For mirrored: show absolute values on y-axis, add direction labels
            if ax == axes[0]:
                ax.set_ylabel("Count", fontsize=FONTSIZE["axis_label"])
            # Make y-tick labels show absolute values
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{abs(int(x))}"))
            # Add text labels for direction
            ax.text(0.02, 0.97, f"Success (n={len(success_preds)})", transform=ax.transAxes,
                    fontsize=FONTSIZE["annotation"], fontweight="bold", color=COLORS["success"], va="top")
            ax.text(0.02, 0.03, f"Failure (n={len(fail_preds)})", transform=ax.transAxes,
                    fontsize=FONTSIZE["annotation"], fontweight="bold", color=COLORS["failure"], va="bottom")
            # No legend needed - labels are inline
        else:
            if ax == axes[0]:
                ax.set_ylabel("Count", fontsize=FONTSIZE["axis_label"])
            ax.legend(fontsize=FONTSIZE["legend"], loc="upper left", framealpha=0.95, edgecolor="none")

        # Add subtle grid
        add_subtle_grid(ax, axis="y")

    plt.tight_layout()
    if show_logos and logos:
        plt.subplots_adjust(top=0.90)

    if save_path:
        _save_figure(fig, save_path)

    return fig


def plot_calibration_curves_by_model(
    model_results: dict[str, dict[str, tuple[ArrayLike, ArrayLike]]],
    num_bins: int = 4,
    save_path: Path | None = None,
    methods_to_show: list[str] | None = None,
    show_logos: bool = True,
) -> plt.Figure:
    """Plot calibration curves for each model showing pre/post/adversarial.

    Creates a 1x3 panel figure showing calibration curves for GPT, Gemini,
    and Claude. Each panel shows pre-execution (coral, solid), post-execution
    (gray, dashed), and adversarial (mint, dotted) methods with distinct
    line styles for better differentiation.

    Args:
        model_results: model -> method -> (predictions, labels)
        num_bins: Number of calibration bins.
        save_path: Optional path to save figure.
        methods_to_show: Methods to include (default: pre, post, adversarial).
        show_logos: If True, show model logos above panel titles.

    Returns:
        Matplotlib figure.
    """
    setup_latex_style()

    # Default methods
    if methods_to_show is None:
        methods_to_show = ["exploration_direct", "review_direct", "review_adversarial"]

    # Model order and display names
    model_order = {"gpt": 0, "gemini": 1, "claude": 2}
    models = sorted(
        model_results.keys(),
        key=lambda m: model_order.get(m.lower().split("-")[0], 99)
    )

    n_models = len(models)
    fig, axes = plt.subplots(1, n_models, figsize=(4.2 * n_models, 4.0))
    if n_models == 1:
        axes = [axes]

    # Load logos
    logos = {}
    if show_logos:
        for model in models:
            model_lower = model.lower()
            if "gpt" in model_lower:
                logo_key = "gpt"
            elif "gemini" in model_lower:
                logo_key = "gemini"
            elif "claude" in model_lower:
                logo_key = "claude"
            else:
                logo_key = None
            if logo_key:
                logo = _load_logo(logo_key, size=120)
                if logo is not None:
                    logos[model] = logo

    for ax, model in zip(axes, models):
        if model not in model_results:
            continue

        method_data = model_results[model]

        # Shade overconfidence region (below diagonal)
        ax.fill_between(
            [0, 1], [0, 1], [0, 0],
            alpha=0.08,
            color="#FF6B6B",
            zorder=0,
            label=None,
        )
        ax.text(
            0.65, 0.25,
            "overconfident",
            fontsize=FONTSIZE["reference_text"],
            color="#CC5555",
            style="italic",
            alpha=0.7,
        )

        # Perfect calibration line (dashed diagonal)
        ax.plot(
            [0, 1], [0, 1],
            color=COLORS["reference"],
            linestyle="--",
            lw=1.5,
            zorder=1,
        )
        ax.text(
            0.82, 0.72,
            "perfect",
            fontsize=FONTSIZE["reference_text"],
            color="#999999",
            rotation=45,
            va="bottom",
            style="italic",
        )

        # Plot each method's calibration curve
        method_order = ["exploration_direct", "review_direct", "review_adversarial"]
        for method in method_order:
            if method not in methods_to_show or method not in method_data:
                continue

            preds, labels = method_data[method]
            bin_means, bin_accuracies, _ = _compute_calibration_bins(
                preds, labels, num_bins
            )
            style = _get_method_style(method)

            # Filter out empty bins
            valid = ~np.isnan(bin_means)
            ax.plot(
                bin_means[valid],
                bin_accuracies[valid],
                lw=style.get("linewidth", LINE_WIDTH),
                linestyle=style.get("linestyle", "-"),
                color=style["color"],
                marker=style["marker"],
                markersize=style["markersize"] + 3,
                markeredgecolor=style["markeredgecolor"],
                markeredgewidth=style["markeredgewidth"],
                label=_format_method_name(method),
                zorder=3,
            )

        # Get model color for title
        model_lower = model.lower()
        if "gpt" in model_lower:
            title_color = COLORS["gpt"]
        elif "gemini" in model_lower:
            title_color = COLORS["gemini"]
        elif "claude" in model_lower:
            title_color = COLORS["claude"]
        else:
            title_color = "black"

        # Add logo to the left of title
        if show_logos and model in logos:
            imagebox = OffsetImage(logos[model], zoom=LOGO_ZOOM)
            imagebox.image.axes = ax
            ab = AnnotationBbox(
                imagebox,
                (0.12, 1.08),
                xycoords="axes fraction",
                frameon=False,
                zorder=10,
            )
            ax.add_artist(ab)
            # Title to the right of logo
            ax.set_title(
                _format_model_name(model),
                fontsize=FONTSIZE["title_with_logo"],
                fontweight="bold",
                color=title_color,
                loc="center",
                pad=10,
            )
        else:
            ax.set_title(
                _format_model_name(model),
                fontsize=FONTSIZE["title"],
                fontweight="bold",
                color=title_color,
            )

        ax.set_xlabel("Mean Predicted Probability", fontsize=FONTSIZE["axis_label"])
        if ax == axes[0]:
            ax.set_ylabel("Fraction of Positives", fontsize=FONTSIZE["axis_label"])
        ax.tick_params(labelsize=FONTSIZE["tick_label"])
        ax.legend(fontsize=FONTSIZE["legend"], loc="upper left", framealpha=0.95, edgecolor="none")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)

        # Add subtle grid
        add_subtle_grid(ax, axis="both")

    plt.tight_layout()
    if show_logos and logos:
        plt.subplots_adjust(top=0.90)

    if save_path:
        _save_figure(fig, save_path)

    return fig


def plot_overconfidence_hero(
    data: dict[str, dict[str, dict[str, float]]],
    save_path: Path | None = None,
    show_logos: bool = True,
) -> plt.Figure:
    """Create single-panel bar chart showing overconfidence.

    Punchy hero figure: shows (mean confidence - base rate) for each model/method.
    This directly visualizes "agentic overconfidence".

    Args:
        data: model -> method -> {"auroc", "ece", "mean_conf", "base_rate"}
              Models: "gpt", "gemini", "claude"
              Methods: "pre", "post", "adv"
        save_path: Optional path to save figure.
        show_logos: If True, show model logos in legend at top.

    Returns:
        Matplotlib figure.
    """
    setup_latex_style()

    # Model and method configuration
    models = ["gpt", "gemini", "claude"]
    methods = ["pre", "post", "adv"]

    model_colors = {
        "gpt": COLORS["gpt"],
        "gemini": COLORS["gemini"],
        "claude": COLORS["claude"],
    }

    model_display = {
        "gpt": "GPT",
        "gemini": "Gemini",
        "claude": "Claude",
    }

    method_display = {
        "pre": "Pre-Exec",
        "post": "Post-Exec",
        "adv": "Adversarial",
    }

    # Create figure with extra space at top for logo legend
    fig, ax = plt.subplots(figsize=(3.6, 2.9))

    bar_width = 0.22
    x = np.arange(len(methods))

    # Load logos for legend
    logos = {}
    if show_logos:
        for model in models:
            logo = _load_logo(model, size=150)
            if logo is not None:
                logos[model] = logo

    # Store bar info for label placement
    bar_info = []

    for model_idx, model in enumerate(models):
        if model not in data:
            continue

        # Compute overconfidence = mean_conf - base_rate
        overconf_values = []
        for method in methods:
            if method in data[model]:
                mean_conf = data[model][method].get("mean_conf", 0)
                base_rate = data[model][method].get("base_rate", 0)
                overconf = mean_conf - base_rate
                overconf_values.append(overconf)
            else:
                overconf_values.append(0)

        positions = x + (model_idx - 1) * bar_width

        bars = ax.bar(
            positions,
            overconf_values,
            bar_width * 0.9,
            label=model_display[model],
            color=model_colors[model],
            edgecolor="black",
            linewidth=0.6,
            zorder=3,
        )

        # Store bar info for label placement
        for bar, val in zip(bars, overconf_values):
            bar_info.append({
                "model": model,
                "x": bar.get_x() + bar.get_width() / 2,
                "y": bar.get_height(),
                "val": val,
            })

    # Add value labels above bars
    for info in bar_info:
        ax.annotate(
            f"{info['val']:.0%}",
            xy=(info["x"], info["y"]),
            xytext=(0, 2),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7,
            fontweight="bold",
            color="#333333",
        )

    # Reference line at 0 (perfect calibration)
    ax.axhline(y=0, color=COLORS["reference"], linestyle="-", lw=1.5, zorder=1)

    # Styling
    ax.set_ylabel("Overconfidence", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([method_display[m] for m in methods], fontsize=10)
    ax.tick_params(labelsize=9)

    # Add subtle grid
    add_subtle_grid(ax, axis="y")

    # Set y-axis limits
    max_val = max(info["val"] for info in bar_info) if bar_info else 0.5
    y_max = max(0.60, max_val + 0.08)
    ax.set_ylim(0, y_max)

    if y_max <= 0.60:
        ax.set_yticks([0, 0.1, 0.2, 0.3, 0.4, 0.5])
        ax.set_yticklabels(["0%", "10%", "20%", "30%", "40%", "50%"])
    else:
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8])
        ax.set_yticklabels(["0%", "20%", "40%", "60%", "80%"])

    # Create custom legend with logos in top-right box (not aligned with bar groups)
    if show_logos and logos:
        # Position legend box in top-right, vertically stacked
        legend_x = 0.98  # Right side
        legend_y_start = 0.97  # Top
        v_spacing = 0.14  # Vertical spacing between entries

        for i, model in enumerate(models):
            if model not in logos:
                continue
            y_pos = legend_y_start - i * v_spacing

            # Add logo
            imagebox = OffsetImage(logos[model], zoom=0.10)
            imagebox.image.axes = ax
            ab = AnnotationBbox(
                imagebox,
                (legend_x - 0.12, y_pos),
                xycoords="axes fraction",
                frameon=False,
                zorder=10,
            )
            ax.add_artist(ab)

            # Add model name to the left of logo
            ax.text(
                legend_x - 0.18,
                y_pos,
                model_display[model],
                transform=ax.transAxes,
                ha="right",
                va="center",
                fontsize=8,
                color=model_colors[model],
                fontweight="bold",
            )
    else:
        # Fallback to standard legend
        ax.legend(
            loc="upper right",
            fontsize=8,
            framealpha=0.95,
            edgecolor="none",
        )

    plt.tight_layout()

    if save_path:
        _save_figure(fig, save_path)

    return fig


# =============================================================================
# Adversarial Shift Decomposition
# =============================================================================


def plot_adversarial_shift_decomposition(
    shift_data: dict[str, dict[str, Any]],
    save_path: Path | str | None = None,
    show_logos: bool = True,
) -> plt.Figure:
    """Show whether adversarial framing shifts estimates uniformly or differentially.

    For each model, plots the mean downward shift on passing vs failing instances.
    A uniform shift (equal bars) means adversarial just lowers all estimates;
    a differential shift (taller fail bar) means it improves discrimination.

    Args:
        shift_data: model_key -> {
            "shift_pass": float,
            "shift_fail": float,
            "shift_pass_se": float,
            "shift_fail_se": float,
            "adv_gap": float,
            "std_gap": float,
            "p_value": float,
        }
        save_path: Optional path to save figure.
        show_logos: If True, show model logos above panel titles.

    Returns:
        Matplotlib figure.
    """
    setup_latex_style()

    model_order = {"gpt": 0, "gemini": 1, "claude": 2}
    models = sorted(
        shift_data.keys(),
        key=lambda m: model_order.get(m.lower().split("-")[0], 99),
    )
    n_models = len(models)

    fig, axes = plt.subplots(1, n_models, figsize=(4.2 * n_models, 4.0), sharey=True)
    if n_models == 1:
        axes = [axes]

    # Load logos
    logos = {}
    if show_logos:
        for model in models:
            model_lower = model.lower()
            if "gpt" in model_lower:
                logo_key = "gpt"
            elif "gemini" in model_lower:
                logo_key = "gemini"
            elif "claude" in model_lower:
                logo_key = "claude"
            else:
                logo_key = None
            if logo_key:
                logo = _load_logo(logo_key, size=120)
                if logo is not None:
                    logos[model] = logo

    model_title_colors = {
        m: COLORS.get(
            "gpt" if "gpt" in m.lower()
            else "gemini" if "gemini" in m.lower()
            else "claude",
            COLORS["reference"],
        )
        for m in models
    }

    bar_width = 0.35

    for ax, model in zip(axes, models):
        d = shift_data[model]
        x = np.array([0, 1])
        shifts = [d["shift_pass"], d["shift_fail"]]
        ses = [d["shift_pass_se"], d["shift_fail_se"]]
        title_color = model_title_colors[model]

        # Use success/failure colors consistent with other figures
        bar_colors = [COLORS["success"], COLORS["failure"]]

        bars = ax.bar(
            x,
            shifts,
            width=bar_width * 2,
            yerr=ses,
            capsize=4,
            color=bar_colors,
            edgecolor="white",
            linewidth=0.8,
            alpha=0.75,
            zorder=3,
            error_kw={"linewidth": 1.2, "capthick": 1.2, "color": "#555555"},
        )

        # Mean shift reference line
        mean_shift = (d["shift_pass"] + d["shift_fail"]) / 2
        ax.axhline(
            y=mean_shift,
            color="#999999",
            linewidth=1.2,
            linestyle="--",
            zorder=2,
            label=f"Mean shift ({mean_shift:.2f})",
        )

        # Significance annotation
        p = d["p_value"]
        if p < 0.05:
            sig_text = f"p={p:.3f}*"
        elif p < 0.1:
            sig_text = f"p={p:.2f}"
        else:
            sig_text = f"p={p:.2f} (n.s.)"

        # Add bracket between bars
        y_max = max(shifts) + max(ses) + 0.02
        ax.plot(
            [0, 0, 1, 1],
            [y_max, y_max + 0.01, y_max + 0.01, y_max],
            color="#555555",
            linewidth=1.0,
            zorder=4,
        )
        ax.text(
            0.5, y_max + 0.015, sig_text,
            ha="center", va="bottom",
            fontsize=FONTSIZE["reference_text"],
            color="#555555",
            zorder=4,
        )

        # Gap annotation
        gap_text = f"Gap: {d['std_gap']:.2f} \u2192 {d['adv_gap']:.2f}"
        ax.text(
            0.5, 0.02, gap_text,
            transform=ax.transAxes,
            ha="center", va="bottom",
            fontsize=FONTSIZE["reference_text"],
            color=title_color,
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                edgecolor=title_color,
                alpha=0.8,
            ),
        )

        ax.set_xticks(x)
        ax.set_xticklabels(["Pass", "Fail"], fontsize=FONTSIZE["tick_label"])

        # Add logo and title (matching other multi-panel figures)
        if show_logos and model in logos:
            imagebox = OffsetImage(logos[model], zoom=LOGO_ZOOM)
            imagebox.image.axes = ax
            ab = AnnotationBbox(
                imagebox,
                (0.12, 1.08),
                xycoords="axes fraction",
                frameon=False,
                clip_on=False,
                zorder=10,
            )
            ax.add_artist(ab)
            ax.set_title(
                _format_model_name(model),
                fontsize=FONTSIZE["title_with_logo"],
                fontweight="bold",
                color=title_color,
                loc="center",
                pad=10,
            )
        else:
            ax.set_title(
                _format_model_name(model),
                fontsize=FONTSIZE["title"],
                fontweight="bold",
                color=title_color,
            )

        if ax == axes[0]:
            ax.set_ylabel(
                "Confidence shift (std $-$ adv)",
                fontsize=FONTSIZE["axis_label"],
            )

        ax.legend(
            fontsize=FONTSIZE["legend"],
            loc="upper left",
            framealpha=0.95,
            edgecolor="none",
        )
        add_subtle_grid(ax, axis="y")

    plt.tight_layout()
    if show_logos and logos:
        plt.subplots_adjust(top=0.90)

    if save_path:
        _save_figure(fig, save_path)

    return fig