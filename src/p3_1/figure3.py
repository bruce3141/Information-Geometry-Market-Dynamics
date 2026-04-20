"""
p3_1.figure3

Phase-plane flow alignment figure for SPY, 1993-present. Two panels:

  Panel (a): Fitted SDHO drift field streamplot, colored by flow speed
             |F(x, y)|, with the slow and fast drift-matrix eigenvectors
             traced as dashed lines. Overdamped-node structure explicit.

  Panel (b): Empirical SPY trajectory density in phase space (log-normalized
             2D histogram), with the fitted drift streamplot overlaid and
             the Lyapunov-predicted 1-sigma and 2-sigma covariance ellipses
             drawn at the analytical orientation angle.

The alignment between the predicted ellipses and the empirical density,
without any fitted angle or scale, is the phase-plane version of the §B.5
four-way consistency battery.

Requires internet access to download SPY via yfinance. Theme handling
matches figures.py: dark mode by default, --light for journal-quality,
--both to generate both.

Usage:
    p3_1_figure3              # dark mode only
    p3_1_figure3 --light      # light mode only (for paper submission)
    p3_1_figure3 --both       # both variants

    python -m p3_1.figure3 --both
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.colors import LogNorm
from scipy.linalg import solve_discrete_lyapunov

from p3_1.phase_space import build_phase_space, fit_sdho


# -----------------------------------------------------------------------------
# Geometry helpers
# -----------------------------------------------------------------------------

def _lyapunov_ellipse(k: float, Omega: float, sigma: float):
    """Return (width, height, angle_deg, Sigma) for the 1-sigma ellipse."""
    A = np.array([[1.0, 1.0], [-k, 1.0 - Omega]])
    Q = np.array([[0.0, 0.0], [0.0, sigma ** 2]])
    Sigma = solve_discrete_lyapunov(A, Q)
    eigvals, eigvecs = np.linalg.eigh(Sigma)
    angle_deg = np.degrees(np.arctan2(eigvecs[1, 1], eigvecs[0, 1]))
    width = 2.0 * np.sqrt(eigvals[1])
    height = 2.0 * np.sqrt(eigvals[0])
    return width, height, angle_deg, Sigma


def _drift_eigenvectors(k: float, Omega: float):
    """Eigenvalues and eigenvectors of the continuous-time drift matrix."""
    A_cont = np.array([[0.0, 1.0], [-k, -Omega]])
    eigvals, eigvecs = np.linalg.eig(A_cont)
    order = np.argsort(np.abs(eigvals))  # slow first
    return eigvals[order], eigvecs[:, order]


# -----------------------------------------------------------------------------
# Themes (independent from figures.py to avoid coupling)
# -----------------------------------------------------------------------------

@dataclass
class _Theme:
    name: str
    bg: str
    fg: str
    grid: str
    density_cm: str
    stream_cm: str
    ellipse_color: str
    slow_color: str
    fast_color: str


def _dark() -> _Theme:
    return _Theme(
        name="dark", bg="#0a0e14", fg="#e6e6e6", grid="#2a3340",
        density_cm="inferno", stream_cm="viridis",
        ellipse_color="#00d4e6", slow_color="#ffd93d", fast_color="#ff7eb6",
    )


def _light() -> _Theme:
    return _Theme(
        name="light", bg="white", fg="#1a1a1a", grid="#d8dde3",
        density_cm="Blues", stream_cm="plasma",
        ellipse_color="#cc3300", slow_color="#005a9c", fast_color="#7a2482",
    )


# -----------------------------------------------------------------------------
# Data fetch
# -----------------------------------------------------------------------------

def _fetch_spy():
    import yfinance as yf
    df = yf.download("SPY", start="1993-01-01", progress=False, auto_adjust=True)
    if df.empty or len(df) < 2000:
        raise RuntimeError(
            f"SPY download returned insufficient data (n={len(df)}). "
            "Check internet connection."
        )
    return np.asarray(df['Close']).ravel()


# -----------------------------------------------------------------------------
# Figure generation
# -----------------------------------------------------------------------------

def figure_3(out_path: str, theme: _Theme):
    """Render Figure 3 at the given theme to out_path."""
    plt.rcParams.update({
        'axes.facecolor':    theme.bg,
        'figure.facecolor':  theme.bg,
        'savefig.facecolor': theme.bg,
        'axes.edgecolor':    theme.fg,
        'axes.labelcolor':   theme.fg,
        'text.color':        theme.fg,
        'xtick.color':       theme.fg,
        'ytick.color':       theme.fg,
        'font.family':       'DejaVu Sans',
        'axes.labelsize':    13,
        'axes.titlesize':    14,
    })

    prices = _fetch_spy()
    x, y, dy = build_phase_space(prices, n=5)
    k, Omega, R2 = fit_sdho(x, y, dy)
    residuals = dy - (-k * x - Omega * y)
    sigma = float(np.std(residuals, ddof=0))

    w1, h1, ang_deg, Sigma = _lyapunov_ellipse(k, Omega, sigma)
    eigvals, eigvecs = _drift_eigenvectors(k, Omega)

    x_std = float(np.std(x)); y_std = float(np.std(y))
    xlim = 3.0 * x_std
    ylim = 3.0 * y_std

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))

    # ---- Panel (a): Fitted SDHO drift field ---------------------------------
    ax = axes[0]
    nx, ny = 40, 40
    XG = np.linspace(-xlim, xlim, nx)
    YG = np.linspace(-ylim, ylim, ny)
    XX, YY = np.meshgrid(XG, YG)
    U = YY
    V = -k * XX - Omega * YY
    speed = np.sqrt(U ** 2 + V ** 2)

    strm = ax.streamplot(XX, YY, U, V, color=speed, cmap=theme.stream_cm,
                         density=1.6, linewidth=1.0, arrowsize=1.2)
    cb = plt.colorbar(strm.lines, ax=ax, pad=0.02, fraction=0.046)
    cb.set_label(r'Flow speed $|F(x, y)|$', color=theme.fg)
    cb.ax.yaxis.set_tick_params(color=theme.fg)
    plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color=theme.fg)

    t_max = max(xlim, ylim) * 1.5
    for eigval, eigvec, color, name in zip(
        eigvals, eigvecs.T, (theme.slow_color, theme.fast_color), ('slow', 'fast')
    ):
        ev = np.real(eigvec)
        ev = ev / np.linalg.norm(ev)
        ax.plot([-t_max * ev[0], t_max * ev[0]],
                [-t_max * ev[1], t_max * ev[1]],
                color=color, linestyle='--', linewidth=1.6, alpha=0.8,
                label=fr'{name} eigenvec ($\lambda = {np.real(eigval):.2f}$)',
                zorder=5)

    ax.set_xlim(-xlim, xlim)
    ax.set_ylim(-ylim, ylim)
    ax.set_xlabel(r'Position $x$')
    ax.set_ylabel(r'Velocity $y = \dot{x}$')
    ax.set_title(f'(a) Fitted SDHO drift field\n'
                 f'$k = {k:.3f}$, $\\Omega = {Omega:.3f}$ (overdamped node)')
    ax.grid(True, color=theme.grid, alpha=0.3, linestyle=':')
    ax.set_axisbelow(True)
    leg = ax.legend(loc='lower right', framealpha=0.8, facecolor=theme.bg,
                    edgecolor=theme.fg, labelcolor=theme.fg, fontsize=10)
    leg.set_zorder(10)

    # ---- Panel (b): Empirical density + Lyapunov ellipses -------------------
    ax = axes[1]
    H, _, _ = np.histogram2d(x, y, bins=80,
                             range=[[-xlim, xlim], [-ylim, ylim]])
    H = H.T
    H_plot = np.ma.masked_where(H == 0, H)
    im = ax.imshow(H_plot, origin='lower', aspect='auto',
                   extent=(-xlim, xlim, -ylim, ylim),
                   cmap=theme.density_cm, norm=LogNorm(vmin=1, vmax=H.max()),
                   interpolation='nearest', alpha=0.95)
    cb = plt.colorbar(im, ax=ax, pad=0.02, fraction=0.046)
    cb.set_label('Count per bin (log)', color=theme.fg)
    cb.ax.yaxis.set_tick_params(color=theme.fg)
    plt.setp(plt.getp(cb.ax.axes, 'yticklabels'), color=theme.fg)

    ax.streamplot(XX, YY, U, V, color=theme.fg, density=1.1,
                  linewidth=0.7, arrowsize=0.9, arrowstyle='->')

    for n_sig, alpha_val in zip((1, 2), (0.9, 0.6)):
        el = Ellipse(xy=(0, 0), width=n_sig * w1, height=n_sig * h1,
                     angle=ang_deg, fill=False, edgecolor=theme.ellipse_color,
                     linewidth=2.4, linestyle='-', alpha=alpha_val, zorder=6)
        ax.add_patch(el)
    ax.plot([], [], color=theme.ellipse_color, linewidth=2.4,
            label='Lyapunov 1-, 2-$\\sigma$ ellipses')
    ax.plot(0, 0, marker='+', color=theme.fg, markersize=12, mew=2, zorder=7,
            label='equilibrium')

    ax.set_xlim(-xlim, xlim)
    ax.set_ylim(-ylim, ylim)
    ax.set_xlabel(r'Position $x$')
    ax.set_ylabel(r'Velocity $y = \dot{x}$')
    ax.set_title(f'(b) SPY empirical density + flow alignment\n'
                 f'$n = {len(x)}$ points, $R^2 = {R2:.3f}$')
    ax.grid(True, color=theme.grid, alpha=0.3, linestyle=':')
    ax.set_axisbelow(True)
    leg = ax.legend(loc='upper right', framealpha=0.8, facecolor=theme.bg,
                    edgecolor=theme.fg, labelcolor=theme.fg, fontsize=10)
    leg.set_zorder(10)

    plt.tight_layout()
    plt.savefig(out_path, dpi=160, facecolor=theme.bg, bbox_inches='tight')
    plt.close()
    print(f"  wrote {out_path}   [k={k:.4f}, Omega={Omega:.4f}, R^2={R2:.4f}]")


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate Figure 3 (phase-plane flow alignment for SPY).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  p3_1_figure3              # dark mode\n"
            "  p3_1_figure3 --light      # light mode (paper submission)\n"
            "  p3_1_figure3 --both       # both variants\n"
        ),
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--light", action="store_true",
                      help="Light-mode (white background) variant only.")
    mode.add_argument("--both", action="store_true",
                      help="Generate both dark-mode and light-mode.")
    parser.add_argument("--out-dir", default="figures",
                        help="Output directory (default: figures/).")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    def _gen(theme: _Theme):
        suffix = "_light" if theme.name == "light" else ""
        out_path = os.path.join(args.out_dir, f"Figure3_phase_flow_alignment{suffix}.png")
        print(f"Generating {theme.name}-mode Figure 3 into {args.out_dir}/...")
        figure_3(out_path, theme)

    if args.both:
        _gen(_dark())
        _gen(_light())
    elif args.light:
        _gen(_light())
    else:
        _gen(_dark())
    print("Done.")


if __name__ == "__main__":
    main()
