"""
analytics/gex_plot.py
---------------------
Interactive Plotly charts for GEXResult objects.

Requires plotly (>=5.0).

Usage
-----
    from analytics.gex import GEXAnalytics
    from analytics.gex_plot import plot_gex, plot_gex_by_expiry

    ga  = GEXAnalytics(ib)
    res = ga.compute(spy_contract)
    plot_gex(res)               # opens in default browser
    plot_gex(res, show=False)   # build figure without rendering
"""

from __future__ import annotations

import sys as _sys
import pathlib as _pathlib

# Ensure project root is on sys.path when the file is run directly
_root = str(_pathlib.Path(__file__).resolve().parent.parent)
if _root not in _sys.path:
    _sys.path.insert(0, _root)

from typing import Optional

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError as _e:  # pragma: no cover
    raise ImportError(
        "plotly is required for GEX plots.  Install with: pip install plotly"
    ) from _e

from analytics.gex import GEXResult


# ── colour palette ──────────────────────────────────────────────────────────
CALL_CLR = "#26a69a"   # teal
PUT_CLR  = "#ef5350"   # red
NET_POS  = "#1976d2"   # blue (positive net)
NET_NEG  = "#e53935"   # red  (negative net)
SPOT_CLR = "#ff9800"   # orange
ZERO_CLR = "#9c27b0"   # purple
WALL_CLR = "#90a4ae"   # blue-grey
BG_DARK  = "#0d0d1a"
PANEL    = "#1a1a2e"


def plot_gex(
    result: GEXResult,
    *,
    show: bool = True,
    title: Optional[str] = None,
) -> "go.Figure":
    """
    Interactive bar chart of GEX by strike, with a cumulative GEX subplot
    and vertical reference lines for spot, zero-gamma, call wall and put wall.

    Parameters
    ----------
    result : GEXResult from GEXAnalytics.compute()
    show   : if True, open the chart in the default browser (default True)
    title  : override the auto-generated title

    Returns
    -------
    plotly.graph_objects.Figure
    """
    profile = result.profile
    if profile.empty:
        raise ValueError("GEXResult.profile is empty - nothing to plot.")

    _B = 1e9
    strikes    = profile["strike"].tolist()
    call_gex_b = (profile["call_gex"] / _B).tolist()
    put_gex_b  = (profile["put_gex"]  / _B).tolist()
    net_gex_b  = (profile["net_gex"]  / _B).tolist()
    cum_b      = (profile["cumulative_gex"] / _B).tolist()

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        row_heights=[0.75, 0.25],
        vertical_spacing=0.04,
        subplot_titles=("", "Cumulative GEX ($B)"),
    )

    # ── top panel: call / put / net bars ────────────────────────────────────
    fig.add_trace(go.Bar(
        x=strikes, y=call_gex_b, name="Call GEX",
        marker_color=CALL_CLR, opacity=0.75,
        hovertemplate="Strike %{x}<br>Call GEX %{y:.3f} B<extra></extra>",
    ), row=1, col=1)

    fig.add_trace(go.Bar(
        x=strikes, y=put_gex_b, name="Put GEX",
        marker_color=PUT_CLR, opacity=0.75,
        hovertemplate="Strike %{x}<br>Put GEX %{y:.3f} B<extra></extra>",
    ), row=1, col=1)

    net_colors = [NET_POS if v >= 0 else NET_NEG for v in net_gex_b]
    fig.add_trace(go.Bar(
        x=strikes, y=net_gex_b, name="Net GEX",
        marker_color=net_colors, opacity=0.95, width=0.35,
        hovertemplate="Strike %{x}<br>Net GEX %{y:.3f} B<extra></extra>",
    ), row=1, col=1)

    # ── bottom panel: cumulative GEX ────────────────────────────────────────
    cum_colors = [NET_POS if v >= 0 else NET_NEG for v in cum_b]
    fig.add_trace(go.Bar(
        x=strikes, y=cum_b, name="Cumulative GEX",
        marker_color=cum_colors, opacity=0.85, showlegend=False,
        hovertemplate="Strike %{x}<br>Cum GEX %{y:.3f} B<extra></extra>",
    ), row=2, col=1)

    # ── vertical reference lines ─────────────────────────────────────────────
    levels = [
        (result.spot,             SPOT_CLR, "solid", 2.2, f"Spot {result.spot:.2f}"),
        (result.zero_gamma_level, ZERO_CLR, "dash",  2.0, f"Gamma Flip {result.zero_gamma_level:.2f}"),
        (result.call_wall,        WALL_CLR, "dot",   1.8, f"Call Wall {result.call_wall:.2f}"),
        (result.put_wall,         PUT_CLR,  "dot",   1.8, f"Put Wall {result.put_wall:.2f}"),
    ]
    for x, colour, dash, width, label in levels:
        if x is None or x <= 0:
            continue
        # top panel: line + inline annotation
        fig.add_vline(
            x=x, line_color=colour, line_dash=dash, line_width=width,
            annotation_text=label,
            annotation_position="top right",
            annotation_font=dict(color=colour, size=10),
            annotation_bgcolor="rgba(13,13,26,0.75)",
            row=1, col=1,
        )
        # bottom panel: line only
        fig.add_vline(
            x=x, line_color=colour, line_dash=dash, line_width=width,
            row=2, col=1,
        )

    _title = title or (
        f"{result.symbol} | GEX Profile | spot={result.spot:.2f}  "
        f"r={result.rate*100:.2f}%  |  total={result.total_gex/_B:.2f}B"
    )

    fig.update_layout(
        title=dict(text=_title, font=dict(color="white", size=15)),
        barmode="overlay",
        template="plotly_dark",
        paper_bgcolor=BG_DARK,
        plot_bgcolor=PANEL,
        bargap=0.05,
        legend=dict(
            orientation="h", y=1.06, x=1, xanchor="right",
            bgcolor="rgba(0,0,0,0)", font=dict(color="white", size=10),
        ),
        margin=dict(l=60, r=30, t=80, b=50),
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="GEX ($B)", row=1, col=1, gridcolor="rgba(255,255,255,0.1)")
    fig.update_yaxes(title_text="Cum ($B)", row=2, col=1, gridcolor="rgba(255,255,255,0.1)")
    fig.update_xaxes(title_text="Strike",   row=2, col=1, gridcolor="rgba(255,255,255,0.1)")
    fig.update_xaxes(showgrid=True, row=1, col=1, gridcolor="rgba(255,255,255,0.1)")

    if show:
        fig.show()
    return fig


def plot_gex_by_expiry(
    result: GEXResult,
    *,
    show: bool = True,
    title: Optional[str] = None,
) -> "go.Figure":
    """
    Horizontal bar chart of net GEX contribution per expiration date.
    """
    by_expiry = result.by_expiry
    if by_expiry.empty:
        raise ValueError("GEXResult.by_expiry is empty - nothing to plot.")

    net_col = "net_gex" if "net_gex" in by_expiry.columns else by_expiry.columns[-1]
    exps    = by_expiry.index.astype(str).tolist()
    values  = (by_expiry[net_col] / 1e9).tolist()
    colors  = [CALL_CLR if v >= 0 else PUT_CLR for v in values]

    fig = go.Figure(go.Bar(
        x=values, y=exps, orientation="h",
        marker_color=colors, opacity=0.9,
        hovertemplate="%{y}<br>Net GEX %{x:.3f} B<extra></extra>",
    ))
    fig.add_vline(x=0, line_color="white", line_width=1)

    _title = title or f"{result.symbol} | Net GEX by Expiration"
    fig.update_layout(
        title=dict(text=_title, font=dict(color="white", size=14)),
        template="plotly_dark",
        paper_bgcolor=BG_DARK,
        plot_bgcolor=PANEL,
        xaxis_title="Net GEX ($B)",
        yaxis_title="Expiration",
        margin=dict(l=80, r=30, t=70, b=50),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.1)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.1)", autorange="reversed")

    if show:
        fig.show()
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Standalone runner — `python -m analytics.gex_plot SPY` (or just SYMBOL arg)
# ─────────────────────────────────────────────────────────────────────────────
def _main(symbol: str = "SPY", expirations: int = 3) -> None:
    """Connect to IB, compute GEX for *symbol* and open both Plotly charts."""
    from ib_async import Stock

    from analytics.gex import GEXAnalytics
    from core.connection import IBConnection

    with IBConnection() as ib:
        contract = Stock(symbol, "SMART", "USD")
        ib.qualifyContracts(contract)
        ga = GEXAnalytics(ib)
        res = ga.compute(contract, expirations=expirations)
        print(res.summary())
        if res.profile.empty:
            print("No GEX profile data available — aborting plot.")
            return
        plot_gex(res)
        plot_gex_by_expiry(res)


if __name__ == "__main__":
    import sys

    sym = sys.argv[1] if len(sys.argv) > 1 else "SPY"
    exps = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    _main(sym, exps)
