import os
import pandas as pd
import streamlit as st
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output"

def get_output_dir() -> Path:
    # Set when launching: $env:PAIRS_OUTPUT_DIR = "output/dev"
    return Path(os.environ.get("PAIRS_OUTPUT_DIR", DEFAULT_OUTPUT_DIR))

OUTPUT_DIR = get_output_dir()

@st.cache_data(ttl=3600)
def load_data(filename: str) -> pd.DataFrame:
    """Load output CSV with spinner."""
    with st.spinner(f"Loading {filename}..."):
        df = pd.read_csv(OUTPUT_DIR / filename)
    st.success(f"Loaded {filename} ({len(df)} rows)")
    return df

summary = load_data("nse_pairs_summary.csv")
trades = load_data("nse_pairs_trades_detailed.csv")
pairs = load_data("cointegrated_pairs.csv")

st.set_page_config(page_title="NSE Pairs Trading — Backtest Dashboard", layout="wide")
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
# Color code Net_PnL
def color_pnl(val):
    if val > 0:
        return 'color: #00a859'
    elif val < 0:
        return 'color: #dc3545'
    return ''

styled_summary = styled_summary.map(color_pnl, subset=["Net_PnL"])
st.dataframe(styled_summary, width='stretch', height=300)

# --- Cointegrated Pairs ---
st.subheader("Cointegrated Pairs")
styled_pairs = pairs.style.format({
    "EG_P": "{:.4f}",
    "Beta": "{:.4f}",
})
st.dataframe(styled_pairs, width='stretch', height=200)

st.markdown("""---""")

# --- Trade Detail with Interactive Pair Selector ---
st.subheader("Trade Detail")
selected_pair = st.selectbox("Select a pair", trades["Pair"].unique())

# Show thinking/processing indicator
with st.status(f"Loading trade data for **{selected_pair}**...", expanded=True) as status:
    pair_trades = trades[trades["Pair"] == selected_pair].sort_values("Date")
    status.update(label="Trade data loaded successfully!", state="complete")

# Interactive Plotly chart instead of st.line_chart
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=pair_trades["Date"],
    y=pair_trades["Cum_PnL"],
    mode='lines+markers',
    line=dict(color='#2196f3', width=2),
    marker=dict(size=4, color='rgba(33,150,243,0.8)'),
    name="Cumulative PnL"
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
    mode='markers',
    marker=dict(symbol='triangle-up', size=10, color='green'),
    name='Entry'
)
fig.add_scatter(
    x=exit_dates, y=exit_vals,
    mode='markers',
    marker=dict(symbol='triangle-down', size=10, color='red'),
    name='Exit'
)

fig.update_layout(
    title=f"Cumulative PnL — {selected_pair}",
    xaxis_title="Date",
    yaxis_title="Cumulative PnL (₹)",
    hovermode='x unified',
    template="plotly_white",
    height=500,
    margin=dict(l=50, r=50, t=80, b=50)
)
st.plotly_chart(fig, width='stretch')

# Trade table with styling
st.dataframe(
    pair_trades.style.format({
        "PnL_Net": "₹{:,.2f}",
        "Cum_PnL": "₹{:,.2f}",
        "Spread": "{:.2f}",
        "ZScore": "{:.2f}",
    }),
    use_container_width=True,
    height=400
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
            x=['Z-Score', 'Lag', 'Vol Ratio', 'Momentum', 'RSI'],
            y=[0.3, 0.25, 0.2, 0.15, 0.1],
            marker_color=['#2196f3', '#ff9800', '#4caf50', '#9c27b0', '#f44336']
        ))
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("Pair analysis data not available")

# --- Footer ---
st.markdown("""""")
st.caption(f"Dashboard generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Data from output directory")