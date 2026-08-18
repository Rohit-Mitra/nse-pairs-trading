import os
import sys
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pairs_trading.backtest import risk_metrics
from src.pairs_trading.config import load_config

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output"


def get_output_dir() -> Path:
    # Set when launching: $env:PAIRS_OUTPUT_DIR = "output/dev"
    return Path(os.environ.get("PAIRS_OUTPUT_DIR", DEFAULT_OUTPUT_DIR))


OUTPUT_DIR = get_output_dir()
CONFIG = load_config(PROJECT_ROOT / "config" / "config.yaml")

st.set_page_config(page_title="NSE Pairs Trading — Backtest Dashboard", layout="wide")


@st.cache_data(ttl=3600)
def load_data(filename: str) -> pd.DataFrame:
    """Load output CSV with spinner."""
    with st.spinner(f"Loading {filename}..."):
        df = pd.read_csv(OUTPUT_DIR / filename)
    st.success(f"Loaded {filename} ({len(df)} rows)")
    return df


def _infer_pred_signal(
    zscore_lag1: pd.Series,
    position: pd.Series,
    entry_z: float,
) -> pd.Series:
    """
    Infer frozen ML entry intent from CSV rows (path b — no model artifact).
    Long-zone bars default to pred_signal=1; short-zone bars to 0.
    """
    pred = pd.Series(0, index=zscore_lag1.index, dtype=int)
    long_zone = zscore_lag1 < -entry_z
    short_zone = zscore_lag1 > entry_z
    pred.loc[long_zone] = 1
    pred.loc[short_zone] = 0
    # Bars where the original run held a long while in the long zone confirm intent.
    pred.loc[long_zone & (position == 1)] = 1
    return pred


def _implied_spread_factor(
    pair_df: pd.DataFrame,
    cap: float,
    txn_cost: float,
) -> pd.Series:
    """
    Back out (y_ret - beta * x_ret) from frozen PnL on days with a lagged position.
    Gap days are filled via Spread pct-change scaling — approximate, not a price re-download.
    """
    d = pair_df.sort_values("Date").copy()
    d["Date"] = pd.to_datetime(d["Date"])
    d = d.set_index("Date")
    lag_pos = d["Position"].shift(1).fillna(0)
    orig_txn = d["Position"].diff().fillna(0).abs() * cap * txn_cost
    pnl_gross = d["PnL_Net"] + orig_txn
    factor = pd.Series(0.0, index=d.index)
    active = lag_pos != 0
    factor.loc[active] = pnl_gross.loc[active] / (lag_pos.loc[active] * cap)
    spread_pct = d["Spread"].pct_change().fillna(0)
    if active.any():
        ratio = (factor.loc[active] / spread_pct.loc[active].replace(0, np.nan)).dropna()
        scale = float(ratio.median()) if len(ratio) else 1.0
        if np.isfinite(scale):
            factor.loc[~active] = spread_pct.loc[~active] * scale
    return factor


def resimulate_from_csv(
    pair_df: pd.DataFrame,
    cap: float,
    base_config,
    entry_z: float,
    exit_z: float,
    txn_cost: float,
) -> tuple[pd.DataFrame, int]:
    """
    Lightweight re-derivation from nse_pairs_trades_detailed.csv — NOT a full ML re-run.

    Re-applies simulate()'s position / PnL rules (backtest.py) using:
    - zscore_lag1 derived from frozen ZScore (shift-1, same as features.py)
    - pred_signal inferred from frozen rows (no saved model/scaler)
    - spread returns implied from frozen PnL + Spread (no price re-download)
    """
    d = pair_df.sort_values("Date").copy()
    d["Date"] = pd.to_datetime(d["Date"])
    d = d.set_index("Date")
    d["zscore_lag1"] = d["ZScore"].shift(1)
    d["pred_signal"] = _infer_pred_signal(d["zscore_lag1"], d["Position"], base_config.entry_z)

    cfg = replace(base_config, entry_z=entry_z, exit_z=exit_z, txn_cost=txn_cost)

    # Mirrors simulate() position logic in src/pairs_trading/backtest.py
    d["position"] = 0
    d.loc[(d["zscore_lag1"] < -cfg.entry_z) & (d["pred_signal"] == 1), "position"] = 1
    d.loc[(d["zscore_lag1"] > cfg.entry_z) & (d["pred_signal"] == 0), "position"] = -1
    d.loc[d["zscore_lag1"].abs() < cfg.exit_z, "position"] = 0
    d["position"] = d["position"].replace(0, np.nan).ffill(limit=cfg.ffill_limit).fillna(0)

    spread_factor = _implied_spread_factor(pair_df, cap, base_config.txn_cost)
    spread_factor = spread_factor.reindex(d.index).fillna(0)

    d["pnl_gross"] = d["position"].shift(1).fillna(0) * cap * spread_factor
    d["txn_cost"] = d["position"].diff().fillna(0).abs() * cap * cfg.txn_cost
    d["pnl_net"] = d["pnl_gross"] - d["txn_cost"]
    d["cum_pnl"] = d["pnl_net"].cumsum()
    n_trades = int((d["position"].diff().fillna(0) != 0).sum())
    return d, n_trades


def _init_resim_session_state() -> None:
    defaults = {
        "resim_applied_pair": None,
        "resim_applied_entry_z": CONFIG.entry_z,
        "resim_applied_exit_z": CONFIG.exit_z,
        "resim_applied_txn_cost": CONFIG.txn_cost,
        "resim_result_df": None,
        "resim_n_trades": 0,
        "resim_metrics": {},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


summary = load_data("nse_pairs_summary.csv")
trades = load_data("nse_pairs_trades_detailed.csv")
pairs = load_data("cointegrated_pairs.csv")

_init_resim_session_state()

# --- Sidebar: Feature 1 parameter inputs (draft; applied only on button click) ---
with st.sidebar:
    st.header("Re-simulate")
    st.caption("Adjust thresholds for the selected pair, then click Re-simulate.")

    draft_entry_z = st.slider(
        "Entry Z",
        min_value=0.5,
        max_value=3.0,
        value=float(CONFIG.entry_z),
        step=0.1,
    )
    exit_z_upper = max(0.0, draft_entry_z - 0.1)
    draft_exit_z = st.slider(
        "Exit Z",
        min_value=0.0,
        max_value=exit_z_upper,
        value=min(float(CONFIG.exit_z), exit_z_upper),
        step=0.1,
    )
    params_valid = draft_exit_z < draft_entry_z
    if not params_valid:
        st.error("Exit Z must be strictly less than Entry Z.")

    draft_txn_cost = st.number_input(
        "Txn cost (fraction of capital)",
        min_value=0.0,
        max_value=0.01,
        value=float(CONFIG.txn_cost),
        step=0.0005,
        format="%.4f",
    )
    resim_clicked = st.button("Re-simulate", disabled=not params_valid)

st.title("NSE Pairs Trading — Backtest Dashboard")

# --- KPI Cards ---
col1, col2, col3, col4 = st.columns(4)
total_pnl = summary["Net_PnL"].sum()
total_return_pct = total_pnl / summary["Capital"].sum() * 100
avg_sharpe = summary["Sharpe"].mean() if "Sharpe" in summary.columns else None
total_trades = summary["Trades"].sum()

col1.metric("Total Net PnL", f"₹{total_pnl:,.0f}", delta=f"{total_return_pct:+.2f}%")
col2.metric("Total Return", f"{total_return_pct:+.2f}%")
col3.metric("Avg Sharpe", f"{avg_sharpe:.2f}" if avg_sharpe is not None else "N/A")
col4.metric("Total Trades", int(total_trades))

st.markdown("""---""")

# --- Pair Summary with Styled Table ---
st.subheader("Pair Summary")
styled_summary = summary.sort_values("Net_PnL", ascending=False).style.format({
    "Net_PnL": "₹{:,.0f}",
    "Return%": "{:+.2f}%",
    "Sharpe": "{:.2f}",
    "TxnCost": "₹{:,.0f}",
    "EndValue": "₹{:,.2f}",
    "Capital": "₹{:,.0f}",
})


def color_pnl(val):
    if val > 0:
        return "color: #00a859"
    if val < 0:
        return "color: #dc3545"
    return ""


styled_summary = styled_summary.map(color_pnl, subset=["Net_PnL"])
st.dataframe(styled_summary, width="stretch", height=300)

# --- Cointegrated Pairs ---
st.subheader("Cointegrated Pairs")
styled_pairs = pairs.style.format({
    "EG_P": "{:.4f}",
    "Beta": "{:.4f}",
})
st.dataframe(styled_pairs, width="stretch", height=200)

st.markdown("""---""")

tab_trade, tab_resim, tab_compare = st.tabs(["Trade Detail", "Re-simulate", "Compare Pairs"])

# --- Trade Detail with Interactive Pair Selector ---
with tab_trade:
    st.subheader("Trade Detail")
    selected_pair = st.selectbox("Select a pair", trades["Pair"].unique(), key="trade_pair_select")

    # Show thinking/processing indicator
    with st.status(f"Loading trade data for **{selected_pair}**...", expanded=True) as status:
        pair_trades = trades[trades["Pair"] == selected_pair].sort_values("Date")
        status.update(label="Trade data loaded successfully!", state="complete")

    # Interactive Plotly chart instead of st.line_chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pair_trades["Date"],
        y=pair_trades["Cum_PnL"],
        mode="lines+markers",
        line=dict(color="#2196f3", width=2),
        marker=dict(size=4, color="rgba(33,150,243,0.8)"),
        name="Cumulative PnL",
    ))

    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

    # Add entry/exit markers based on position changes
    positions = pair_trades["Position"].diff().fillna(0)
    entry_mask = positions == 1
    exit_mask = positions == -1

    entry_dates = pair_trades.loc[entry_mask, "Date"]
    entry_vals = pair_trades.loc[entry_mask, "Cum_PnL"]
    exit_dates = pair_trades.loc[exit_mask, "Date"]
    exit_vals = pair_trades.loc[exit_mask, "Cum_PnL"]

    fig.add_scatter(
        x=entry_dates, y=entry_vals,
        mode="markers",
        marker=dict(symbol="triangle-up", size=10, color="green"),
        name="Entry",
    )
    fig.add_scatter(
        x=exit_dates, y=exit_vals,
        mode="markers",
        marker=dict(symbol="triangle-down", size=10, color="red"),
        name="Exit",
    )

    fig.update_layout(
        title=f"Cumulative PnL — {selected_pair}",
        xaxis_title="Date",
        yaxis_title="Cumulative PnL (₹)",
        hovermode="x unified",
        template="plotly_white",
        height=500,
        margin=dict(l=50, r=50, t=80, b=50),
    )
    st.plotly_chart(fig, width="stretch")

    # Trade table with styling
    st.dataframe(
        pair_trades.style.format({
            "PnL_Net": "₹{:,.2f}",
            "Cum_PnL": "₹{:,.2f}",
            "Spread": "{:.2f}",
            "ZScore": "{:.2f}",
        }),
        use_container_width=True,
        height=400,
    )

    # --- Pair Details Expander ---
    st.markdown("""---""")
    st.subheader("Pair Analysis")
    with st.expander("View pair selection logic & statistics", expanded=False):
        # Find the pair info from summary
        pair_row = summary[summary["Pair"] == selected_pair].iloc[0] if not summary[summary["Pair"] == selected_pair].empty else None

        if pair_row is not None:
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"**Beta:** {pair_row['Beta']:.4f}")
                st.markdown(f"**Window:** {pair_row['Window']}")
                st.markdown(f"**Sharpe:** {pair_row['Sharpe']:.2f}")
            with col_b:
                st.markdown(f"**Trades:** {int(pair_row['Trades'])}")
                st.markdown(f"**Return:** {pair_row['Return%']:.2f}%")
                st.markdown(f"**Max DD:** {abs(pair_row['MaxDD']):,.2f}")

            st.markdown(f"**Capital Deployed:** ₹{pair_row['Capital']:,.0f}")
            st.markdown(f"**Profit Factor:** {pair_row['ProfitFactor']:.3f}")
            st.markdown(f"**Hit Rate:** {pair_row['HitRate']:.2%}")

            # Walk-forward accuracy badge
            wf_acc = pair_row["WF_Acc"]
            st.markdown(f"**Walk-Forward Acc:** {wf_acc:.2%}")
            if wf_acc > 0.6:
                st.success("Strong strategy")
            elif wf_acc > 0.5:
                st.info("Moderate strategy")
            else:
                st.warning("Weak strategy")

            # Feature importance placeholder
            st.markdown("---")
            st.markdown("**Feature Importance** (placeholder for model analysis)")
            fig_bar = go.Figure(go.Bar(
                x=["Z-Score", "Lag", "Vol Ratio", "Momentum", "RSI"],
                y=[0.3, 0.25, 0.2, 0.15, 0.1],
                marker_color=["#2196f3", "#ff9800", "#4caf50", "#9c27b0", "#f44336"],
            ))
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("Pair analysis data not available")

# --- Feature 1: Re-simulate tab ---
with tab_resim:
    st.subheader("Re-simulate Selected Pair")
    resim_pair = st.session_state.get("trade_pair_select", trades["Pair"].unique()[0])
    st.markdown(f"**Pair:** {resim_pair} *(change pair on the Trade Detail tab)*")

    pair_trades_resim = trades[trades["Pair"] == resim_pair].sort_values("Date")
    pair_summary = summary[summary["Pair"] == resim_pair]
    if pair_summary.empty:
        st.warning(f"No summary row found for {resim_pair}.")
    else:
        cap = float(pair_summary.iloc[0]["Capital"])

        if resim_clicked and params_valid:
            result_df, n_trades = resimulate_from_csv(
                pair_trades_resim,
                cap,
                CONFIG,
                draft_entry_z,
                draft_exit_z,
                draft_txn_cost,
            )
            st.session_state.resim_applied_pair = resim_pair
            st.session_state.resim_applied_entry_z = draft_entry_z
            st.session_state.resim_applied_exit_z = draft_exit_z
            st.session_state.resim_applied_txn_cost = draft_txn_cost
            st.session_state.resim_result_df = result_df
            st.session_state.resim_n_trades = n_trades
            st.session_state.resim_metrics = risk_metrics(result_df["pnl_net"], result_df["cum_pnl"])

        has_applied = (
            st.session_state.resim_result_df is not None
            and st.session_state.resim_applied_pair == resim_pair
        )

        if has_applied and st.session_state.resim_n_trades == 0:
            st.warning(
                "Re-simulation produced zero trades — entry/exit thresholds may be too strict "
                "for this pair's z-score path. Try lowering Entry Z or raising Exit Z."
            )
        elif has_applied:
            orig_row = pair_summary.iloc[0]
            orig_net_pnl = float(orig_row["Net_PnL"])
            orig_sharpe = float(orig_row["Sharpe"])
            orig_trades = int(orig_row["Trades"])

            resim_df = st.session_state.resim_result_df
            resim_net_pnl = float(resim_df["pnl_net"].sum())
            resim_sharpe = st.session_state.resim_metrics.get("Sharpe", 0)
            resim_trades = st.session_state.resim_n_trades

            orig_dates = pair_trades_resim["Date"]
            orig_cum = pair_trades_resim["Cum_PnL"]
            resim_dates = resim_df.index.strftime("%Y-%m-%d")
            resim_cum = resim_df["cum_pnl"]

            y_min = min(orig_cum.min(), resim_cum.min())
            y_max = max(orig_cum.max(), resim_cum.max())
            y_pad = (y_max - y_min) * 0.05 if y_max != y_min else 1.0

            chart_left, chart_right = st.columns(2)
            with chart_left:
                fig_orig = go.Figure()
                fig_orig.add_trace(go.Scatter(
                    x=orig_dates,
                    y=orig_cum,
                    mode="lines",
                    name="Original (frozen CSV)",
                    line=dict(color="#2196f3", width=2),
                ))
                fig_orig.update_layout(
                    title="Original cumulative PnL",
                    xaxis_title="Date",
                    yaxis_title="Cumulative PnL (₹)",
                    yaxis=dict(range=[y_min - y_pad, y_max + y_pad]),
                    template="plotly_white",
                    height=420,
                    showlegend=True,
                )
                st.plotly_chart(fig_orig, use_container_width=True)

            with chart_right:
                fig_resim = go.Figure()
                fig_resim.add_trace(go.Scatter(
                    x=resim_dates,
                    y=resim_cum,
                    mode="lines",
                    name="Re-simulated",
                    line=dict(color="#ff9800", width=2),
                ))
                fig_resim.update_layout(
                    title=(
                        f"Re-simulated (entry={st.session_state.resim_applied_entry_z:.1f}, "
                        f"exit={st.session_state.resim_applied_exit_z:.1f}, "
                        f"txn={st.session_state.resim_applied_txn_cost:.4f})"
                    ),
                    xaxis_title="Date",
                    yaxis_title="Cumulative PnL (₹)",
                    yaxis=dict(range=[y_min - y_pad, y_max + y_pad]),
                    template="plotly_white",
                    height=420,
                    showlegend=True,
                )
                st.plotly_chart(fig_resim, use_container_width=True)

            delta_df = pd.DataFrame({
                "Metric": ["Net PnL (₹)", "Sharpe", "Trades"],
                "Original": [orig_net_pnl, orig_sharpe, orig_trades],
                "Re-simulated": [resim_net_pnl, resim_sharpe, resim_trades],
                "Delta": [
                    resim_net_pnl - orig_net_pnl,
                    resim_sharpe - orig_sharpe,
                    resim_trades - orig_trades,
                ],
            })
            st.markdown("**Original vs re-simulated**")
            st.dataframe(
                delta_df.style.format({
                    "Original": "{:,.3f}",
                    "Re-simulated": "{:,.3f}",
                    "Delta": "{:+,.3f}",
                }),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info(
                "Use the sidebar sliders to set Entry Z, Exit Z, and Txn cost, "
                "then click **Re-simulate** to compare against the frozen pipeline output."
            )

# --- Feature 2: Compare Pairs tab ---
with tab_compare:
    st.subheader("Compare Pairs")
    compare_colors = ["#2196f3", "#ff9800", "#4caf50", "#9c27b0"]
    all_pair_names = sorted(trades["Pair"].unique())
    selected_compare = st.multiselect(
        "Select pairs to compare (2–4)",
        options=all_pair_names,
        max_selections=4,
    )

    if len(selected_compare) > 4:
        st.warning("Maximum 4 pairs — showing the first 4 selected.")
        selected_compare = selected_compare[:4]

    if len(selected_compare) < 2:
        st.info("Select at least 2 pairs to overlay cumulative PnL curves.")
    else:
        compare_fig = go.Figure()
        for idx, pair_name in enumerate(selected_compare):
            pair_curve = trades[trades["Pair"] == pair_name].sort_values("Date")
            compare_fig.add_trace(go.Scatter(
                x=pair_curve["Date"],
                y=pair_curve["Cum_PnL"],
                mode="lines",
                name=pair_name,
                line=dict(color=compare_colors[idx % len(compare_colors)], width=2),
            ))
        compare_fig.update_layout(
            title="Cumulative PnL — pair comparison",
            xaxis_title="Date",
            yaxis_title="Cumulative PnL (₹)",
            hovermode="x unified",
            template="plotly_white",
            height=500,
            legend=dict(title="Pair"),
        )
        st.plotly_chart(compare_fig, use_container_width=True)

        compare_table = summary[summary["Pair"].isin(selected_compare)][
            ["Pair", "Sharpe", "MaxDD", "HitRate", "Return%", "Trades"]
        ].copy()
        compare_table = compare_table.set_index("Pair").loc[selected_compare].reset_index()
        st.dataframe(
            compare_table.style.format({
                "Sharpe": "{:.3f}",
                "MaxDD": "₹{:,.2f}",
                "HitRate": "{:.2%}",
                "Return%": "{:+.2f}%",
                "Trades": "{:.0f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

# --- Footer ---
st.markdown("""""")
st.caption(f"Dashboard generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Data from output directory")
