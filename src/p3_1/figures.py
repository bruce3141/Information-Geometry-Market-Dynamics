"""
p3_1.figures

Generate all paper figures (Figure 1a, 1b, and 2) for Dean (2026).

Figure 1a: Pareto frontier with all 23 markets, full Omega range [0, 2.5].
Figure 1b: Detail view of cluster region with bootstrap error bars.
Figure 2:  Same detail window as Figure 1b but with the exact closed form
           R^2_exact(k, Omega) replacing the contact-geometric form.

Two theme variants are supported:

  Dark mode  (default): high-saturation neon palette on near-black background.
                        Optimized for screen viewing and social media.
  Light mode (--light): journal-quality desaturated palette on white
                        background. Recommended for print and PDF submission.

Light-mode files get a `_light` suffix so both variants can coexist:
  Figure1a_pareto_frontier.png         (dark, default)
  Figure1a_pareto_frontier_light.png   (light, journal)

Usage:
    p3_1_figures              # dark mode, default
    p3_1_figures --light      # light mode for journal submission
    p3_1_figures --both       # generate both variants

    python -m p3_1.figures
    python -m p3_1.figures --light
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt

from p3_1.phase_space import r2_exact

# ---- Market data (from Table 1 and Table B1 of P3-1) ----
FUTURES = {  # [1] Dean (2025), Table 2 -- (Omega, R^2)
    "ES": (1.150, 0.570), "NQ": (1.210, 0.590), "CL": (1.210, 0.590),
    "GC": (1.120, 0.560), "ZB": (1.150, 0.580), "6E": (1.140, 0.570),
}
ASIAN = {  # Appendix B Table B1 -- (Omega, R^2, R^2_se, Omega_se)
    "HSI":   (1.092, 0.549, 0.008, 0.017),
    "AXJO":  (1.157, 0.574, 0.008, 0.017),
    "N225":  (1.142, 0.568, 0.008, 0.017),
    "KOSPI": (1.099, 0.551, 0.008, 0.017),
    "TSEC":  (1.058, 0.535, 0.008, 0.016),
}
EUROPEAN = {
    "FTSE":  (1.078, 0.544, 0.008, 0.017),
    "DAX":   (1.074, 0.542, 0.008, 0.016),
    "CAC":   (1.093, 0.549, 0.008, 0.016),
    "STOXX": (1.106, 0.555, 0.008, 0.016),
}
EMERGING = {
    "BVSP":  (1.147, 0.570, 0.008, 0.017),
    "NIFTY": (1.070, 0.539, 0.008, 0.016),
    "IPC":   (1.093, 0.550, 0.008, 0.016),
    "JKSE":  (1.064, 0.538, 0.008, 0.016),
}
CRYPTO = {
    "BTC": (1.154, 0.572, 0.008, 0.016),
    "ETH": (1.177, 0.581, 0.009, 0.019),
    "BNB": (1.132, 0.563, 0.009, 0.019),
    "XRP": (1.151, 0.571, 0.009, 0.019),
}

K_MEAN = 0.223
K_RANGE = (0.197, 0.239)


# =============================================================================
# Theme system
# =============================================================================

@dataclass(frozen=True)
class Theme:
    """Color palette and visual styling for a figure variant.

    Light mode is journal-quality: desaturated colors, white background,
    near-black text, light grey grid. Dark mode is screen-optimized: high-
    saturation neon colors against near-black background.
    """
    name: str
    bg: str
    text: str
    edge: str
    grid: str
    curve: str
    det_fill: str
    stoch_fill: str
    det_label: str
    stoch_label: str
    callout: str
    legend_bg: str
    band_color: str
    marker_futures: str
    marker_asian: str
    marker_european: str
    marker_emerging: str
    marker_crypto: str
    marker_edge: str


def _dark_theme() -> Theme:
    return Theme(
        name="dark",
        bg="#0a0e14",
        text="#e6e6e6",
        edge="#e6e6e6",
        grid="#2a3340",
        curve="#00d4e6",
        det_fill="#0f4c4c",
        stoch_fill="#3a0a0a",
        det_label="#7ff0ff",
        stoch_label="#ff7a7a",
        callout="#ffd93d",
        legend_bg="#0f1620",
        band_color="#00d4e6",
        marker_futures="#ffd93d",
        marker_asian="#ff7eb6",
        marker_european="#8ab4ff",
        marker_emerging="#7fd97f",
        marker_crypto="#c48cff",
        marker_edge="black",
    )


def _light_theme() -> Theme:
    return Theme(
        name="light",
        bg="white",
        text="#1a1a1a",
        edge="#1a1a1a",
        grid="#d8dde3",
        curve="#0066cc",
        det_fill="#e8f4ec",
        stoch_fill="#fce8e8",
        det_label="#1a6e2a",
        stoch_label="#a31515",
        callout="#7a4f00",
        legend_bg="white",
        band_color="#5a8fd6",
        marker_futures="#d4a017",
        marker_asian="#cc3366",
        marker_european="#1f4eb6",
        marker_emerging="#2c8a3a",
        marker_crypto="#7332a3",
        marker_edge="black",
    )


def _setup_theme(theme: Theme):
    plt.rcParams.update({
        "figure.facecolor":  theme.bg,
        "axes.facecolor":    theme.bg,
        "savefig.facecolor": theme.bg,
        "text.color":        theme.text,
        "axes.labelcolor":   theme.text,
        "xtick.color":       theme.text,
        "ytick.color":       theme.text,
        "axes.edgecolor":    theme.edge,
        "axes.titlecolor":   theme.text,
        "font.family":       "DejaVu Sans",
        "font.size":         14,
        "axes.titlesize":    18,
        "axes.labelsize":    16,
        "legend.fontsize":   12,
    })


# =============================================================================
# Figure helpers
# =============================================================================

def _unpack_omega_r2(d: dict):
    arr = np.array([(v[0], v[1]) for v in d.values()])
    return arr[:, 0], arr[:, 1]


def _unpack_with_errors(d: dict):
    arr = np.array(list(d.values()))
    return arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]


def _scatter_markets(ax, theme: Theme, with_errors: bool = False):
    common = dict(edgecolor=theme.marker_edge, linewidth=0.8, zorder=6)

    xf, yf = _unpack_omega_r2(FUTURES)
    ax.scatter(xf, yf, s=130, marker="o", c=theme.marker_futures,
               label="Dean (2025) futures ($n=6$)", **common)

    sets = [
        (ASIAN,    "s", theme.marker_asian,    "Asian equities ($n=5$)",          95),
        (EUROPEAN, "^", theme.marker_european, "European equities ($n=4$)",       110),
        (EMERGING, "D", theme.marker_emerging, "Emerging-market equities ($n=4$)", 110),
        (CRYPTO,   "P", theme.marker_crypto,   "Cryptocurrencies ($n=4$)",        130),
    ]
    for d, marker, color, label, size in sets:
        xs, ys, yerr, xerr = _unpack_with_errors(d)
        if with_errors:
            err_alpha = 0.45 if theme.name == "light" else 0.5
            ax.errorbar(xs, ys, xerr=xerr, yerr=yerr, fmt="none",
                        ecolor=color, alpha=err_alpha, capsize=3, lw=1.2, zorder=5)
        ax.scatter(xs, ys, s=size, marker=marker, c=color, label=label, **common)


def _legend_for_theme(ax, theme: Theme, loc="lower right"):
    legend = ax.legend(loc=loc, framealpha=0.92,
                       facecolor=theme.legend_bg, edgecolor=theme.edge)
    for t in legend.get_texts():
        t.set_color(theme.text)
    return legend


# =============================================================================
# Figure 1a
# =============================================================================

def figure_1a(out_path: str, theme: Theme):
    _setup_theme(theme)
    fig, ax = plt.subplots(figsize=(13, 8.5))

    Omega_grid = np.linspace(0, 2.5, 600)
    R2_curve = Omega_grid ** 2 / (1 + Omega_grid ** 2)
    ax.fill_between(Omega_grid, R2_curve, 1.0, color=theme.stoch_fill, zorder=0)
    ax.fill_between(Omega_grid, 0.0, R2_curve, color=theme.det_fill, zorder=0)
    ax.plot(Omega_grid, R2_curve, color=theme.curve, lw=3.2,
            label=r"Pareto frontier  $R^2 = \Omega^2/(1+\Omega^2)$", zorder=5)

    _scatter_markets(ax, theme, with_errors=False)

    label_kw = dict(fontsize=18, fontweight="bold", ha="center", va="center")
    # Region semantics: above the curve = deterministic-dominated (R^2 exceeds
    # what a stationary linear SDHO permits at that Omega); below the curve =
    # stochastic-dominated (more noise than SDHO permits). The curve itself is
    # the SDHO locus. The `stoch_*` and `det_*` theme fields retain their
    # original names for minimum-diff and should be read as above-fill/label
    # and below-fill/label respectively. Detailed at-market-k interpretation
    # is in §C.4; the empirical battery confirming all 23 markets sit on the
    # curve is in §B.5 Table B3.
    ax.text(2.05, 0.95, "DETERMINISTIC\nDOMINATED", color=theme.stoch_label, **label_kw)
    ax.text(2.05, 0.65, "STOCHASTIC\nDOMINATED", color=theme.det_label, **label_kw)

    callout = (
        "23 markets cluster here\n"
        r"$\Omega \approx 1.12,\ R^2 \approx 0.56$" + "\n"
        r"CV$(\Omega) \approx 3.8\%$,  CV$(R^2) \approx 2.9\%$"
    )
    ax.annotate(
        callout, xy=(1.14, 0.565), xytext=(0.32, 0.84),
        fontsize=13, color=theme.callout, ha="center", va="center",
        bbox=dict(boxstyle="round,pad=0.5,rounding_size=0.25",
                  facecolor=theme.bg, edgecolor=theme.callout, linewidth=1.8),
        arrowprops=dict(arrowstyle="->", color=theme.callout, lw=1.8,
                        connectionstyle="arc3,rad=-0.35", shrinkA=2, shrinkB=6),
        zorder=10,
    )

    ax.set_xlim(0, 2.5)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel(r"Dissipation rate $\Omega$")
    ax.set_ylabel(r"Deterministic fraction $R^2$")
    ax.set_title("The Pareto Frontier of Market Dynamics", fontweight="bold", pad=14)
    ax.grid(True, color=theme.grid, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    _legend_for_theme(ax, theme)

    plt.tight_layout()
    plt.savefig(out_path, dpi=160, facecolor=theme.bg)
    plt.close()
    print(f"  wrote {out_path}")


# =============================================================================
# Figure 1b
# =============================================================================

def figure_1b(out_path: str, theme: Theme):
    _setup_theme(theme)
    fig, ax = plt.subplots(figsize=(11, 8))

    Omega_grid = np.linspace(1.0, 1.25, 400)
    R2_curve = Omega_grid ** 2 / (1 + Omega_grid ** 2)
    ax.plot(Omega_grid, R2_curve, color=theme.curve, lw=3.0,
            label=r"Contact-geometric: $R^2 = \Omega^2/(1+\Omega^2)$", zorder=5)

    _scatter_markets(ax, theme, with_errors=True)

    ax.set_xlim(1.04, 1.22)
    ax.set_ylim(0.52, 0.61)
    ax.set_xlabel(r"Dissipation rate $\Omega$")
    ax.set_ylabel(r"Deterministic fraction $R^2$")
    ax.set_title("Cluster Region Detail (with bootstrap $\\pm 1\\sigma$)",
                 fontweight="bold", pad=14)
    ax.grid(True, color=theme.grid, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    _legend_for_theme(ax, theme)

    plt.tight_layout()
    plt.savefig(out_path, dpi=160, facecolor=theme.bg)
    plt.close()
    print(f"  wrote {out_path}")


# =============================================================================
# Figure 2
# =============================================================================

def figure_2(out_path: str, theme: Theme):
    _setup_theme(theme)
    fig, ax = plt.subplots(figsize=(11, 8))

    Omega_grid = np.linspace(1.04, 1.22, 400)
    R2_exact_mean = np.array([r2_exact(K_MEAN, O) for O in Omega_grid])
    R2_exact_low  = np.array([r2_exact(K_RANGE[0], O) for O in Omega_grid])
    R2_exact_high = np.array([r2_exact(K_RANGE[1], O) for O in Omega_grid])

    band_alpha = 0.25 if theme.name == "light" else 0.18
    ax.fill_between(Omega_grid, R2_exact_low, R2_exact_high,
                    color=theme.band_color, alpha=band_alpha, zorder=4,
                    label=rf"$k \in [{K_RANGE[0]:.3f}, {K_RANGE[1]:.3f}]$")

    ax.plot(Omega_grid, R2_exact_mean, color=theme.curve, lw=3.0,
            label=rf"$R^2_{{\rm exact}}(k = {K_MEAN}, \Omega)$ from eq. (A.1)",
            zorder=5)

    _scatter_markets(ax, theme, with_errors=True)

    ax.set_xlim(1.04, 1.22)
    ax.set_ylim(0.52, 0.61)
    ax.set_xlabel(r"Dissipation rate $\Omega$")
    ax.set_ylabel(r"Deterministic fraction $R^2$")
    ax.set_title("Cluster Region with Exact Discrete-Time Frontier",
                 fontweight="bold", pad=14)
    ax.grid(True, color=theme.grid, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)
    _legend_for_theme(ax, theme)

    plt.tight_layout()
    plt.savefig(out_path, dpi=160, facecolor=theme.bg)
    plt.close()
    print(f"  wrote {out_path}")


# =============================================================================
# Driver
# =============================================================================

def _generate(theme: Theme, out_dir: str):
    suffix = "_light" if theme.name == "light" else ""
    print(f"Generating {theme.name}-mode paper figures into {out_dir}/...")
    figure_1a(os.path.join(out_dir, f"Figure1a_pareto_frontier{suffix}.png"), theme)
    figure_1b(os.path.join(out_dir, f"Figure1b_pareto_frontier_detail{suffix}.png"), theme)
    figure_2(os.path.join(out_dir, f"Figure2_exact_pareto{suffix}.png"), theme)


def main():
    """Generate all paper figures into the figures/ directory.

    Defaults to dark mode (screen-optimized). Pass --light for the journal-
    quality light-mode variant, or --both to generate both.
    """
    parser = argparse.ArgumentParser(
        description="Generate the P3-1 paper figures.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  p3_1_figures              # dark mode (default), screen-optimized\n"
            "  p3_1_figures --light      # light mode, for journal submission\n"
            "  p3_1_figures --both       # both variants\n"
        ),
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--light", action="store_true",
                      help="Generate light-mode (white background) figures only.")
    mode.add_argument("--both", action="store_true",
                      help="Generate both dark-mode and light-mode figures.")
    parser.add_argument("--out-dir", default="figures",
                        help="Output directory (default: figures/).")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    if args.both:
        _generate(_dark_theme(), args.out_dir)
        _generate(_light_theme(), args.out_dir)
    elif args.light:
        _generate(_light_theme(), args.out_dir)
    else:
        _generate(_dark_theme(), args.out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
