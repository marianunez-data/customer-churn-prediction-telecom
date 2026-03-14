"""
Page 2 — ROI Simulator
Interactive sliders: θ, Cr, Vc, Capacity. Real-time profit curve + confusion matrix in $.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from sklearn.metrics import confusion_matrix

st.set_page_config(page_title="ROI Simulator — Churn Intelligence",
                   page_icon="💰", layout="wide")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=DM+Sans:wght@300;400;600;700&display=swap');
:root{--bg-deep:#0A0F1E;--bg-card:#111827;--bg-card2:#1a2236;
    --cyan:#00D4FF;--red:#E84855;--green:#2ECC71;--yellow:#F4C430;
    --text:#E8EDF5;--muted:#8A9BB8;--border:#1E2D45;}
html,body,[data-testid="stAppViewContainer"]{background:var(--bg-deep)!important;
    color:var(--text)!important;font-family:'DM Sans',sans-serif;}
[data-testid="stSidebar"]{background:#0D1321!important;border-right:1px solid var(--border);}
[data-testid="stSidebar"] *{color:var(--text)!important;}
h1,h2,h3,h4{font-family:'DM Sans',sans-serif;color:var(--text)!important;}
.stSlider label p{color:var(--text)!important;}
.page-header{background:linear-gradient(135deg,#0D1321,#111827);border:1px solid var(--border);
    border-left:4px solid var(--yellow);border-radius:10px;padding:26px 34px;margin-bottom:26px;}
.section-card{background:var(--bg-card);border:1px solid var(--border);
    border-radius:10px;padding:22px 26px;margin-bottom:16px;}
.section-title{font-size:10px;letter-spacing:2px;text-transform:uppercase;
    color:var(--cyan);font-family:'IBM Plex Mono',monospace;margin-bottom:14px;}
.kpi-row{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-bottom:16px;}
.kpi{background:var(--bg-card2);border:1px solid var(--border);border-radius:10px;
    padding:18px 22px;text-align:center;position:relative;overflow:hidden;}
.kpi::after{content:'';position:absolute;bottom:0;left:0;right:0;height:2px;}
.kpi.green::after{background:var(--green);}
.kpi.cyan::after{background:var(--cyan);}
.kpi.yellow::after{background:var(--yellow);}
.kpi-val{font-size:28px;font-weight:700;font-family:'IBM Plex Mono',monospace;line-height:1;}
.kpi-lbl{font-size:10px;letter-spacing:2px;text-transform:uppercase;color:var(--muted);
    font-family:'IBM Plex Mono',monospace;margin-top:5px;}
.kpi-sub{font-size:11px;color:var(--muted);margin-top:3px;}
.param-fixed{background:var(--bg-card2);border:1px solid var(--border);border-radius:8px;
    padding:12px 16px;margin-bottom:8px;display:flex;justify-content:space-between;align-items:center;}
.param-name{font-size:11px;color:var(--muted);font-family:'IBM Plex Mono',monospace;}
.param-val{font-size:15px;font-weight:700;font-family:'IBM Plex Mono',monospace;color:var(--cyan);}
.warn{background:rgba(244,196,48,0.08);border:1px solid rgba(244,196,48,0.3);
    border-radius:8px;padding:10px 14px;font-size:11px;color:#F4C430;
    font-family:'IBM Plex Mono',monospace;margin-bottom:8px;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

PROJECT_ROOT = Path(__file__).parent.parent.parent

# ── Load data ─────────────────────────────────────────────────────────────
@st.cache_data
def load_predictions():
    df = pd.read_csv(PROJECT_ROOT / 'reports' / 'final_results_test.csv')
    return df

df_test = load_predictions()

y_true = df_test['y_true'].values
y_cal  = df_test['y_cal'].values
n      = len(y_true)
n_pos  = int(y_true.sum())

# ── Header ────────────────────────────────────────────────────────────────
st.markdown("""
<div class='page-header'>
    <div style='font-size:10px;letter-spacing:3px;text-transform:uppercase;
                color:#F4C430;font-family:IBM Plex Mono,monospace;margin-bottom:8px;'>💰 ROI Simulator</div>
    <div style='font-size:24px;font-weight:700;margin-bottom:6px;'>Business Impact Explorer</div>
    <div style='color:#6B7A99;font-size:13px;'>
        Adjust threshold, costs and capacity to explore real-time ROI on the test cohort (705 customers).
    </div>
</div>
""", unsafe_allow_html=True)

# ── Layout: sliders | outputs ─────────────────────────────────────────────
col_ctrl, col_out = st.columns([1, 2], gap="large")

with col_ctrl:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Business Parameters</div>",
                unsafe_allow_html=True)

    theta = st.slider("Decision Threshold (θ)",
                      min_value=0.05, max_value=0.95,
                      value=0.41, step=0.01,
                      help="Minimum churn probability to flag a customer for intervention.")

    Cr = st.slider("Intervention Cost — Cr ($)",
                   min_value=10, max_value=150,
                   value=20, step=5,
                   help="Cost per contacted customer (agent call). "
                        "Set higher to include retention discounts.")

    Vc = st.slider("Retention Value — Vc ($) ⚠",
                   min_value=20, max_value=200,
                   value=69, step=5,
                   help="Value of a correctly retained customer. "
                        "Default = ARPU×4mo×churn_rate (dataset-grounded).")

    if Vc != 69:
        st.markdown("""
        <div class='warn'>
            ⚠ Vc modified from dataset-grounded default ($69).<br>
            Ensure new value has a documented business justification.
        </div>
        """, unsafe_allow_html=True)

    capacity = st.slider("Call Center Capacity (%)",
                         min_value=10, max_value=50,
                         value=25, step=5,
                         help="Maximum % of the customer base that can be contacted.")

    st.markdown("<hr style='border-color:#1E2D45;margin:14px 0;'>",
                unsafe_allow_html=True)

    # Fixed parameters
    st.markdown("<div class='section-title'>Fixed Parameters</div>",
                unsafe_allow_html=True)
    for label, val in [("ARPU", "$64.76/month"),
                       ("Evidence window", "4 months"),
                       ("Break-even precision", f"{Cr/Vc:.1%}"),
                       ("Test cohort size", f"{n:,} customers"),
                       ("True churners", f"{n_pos} ({n_pos/n:.1%})")]:
        st.markdown(f"""
        <div class='param-fixed'>
            <span class='param-name'>{label}</span>
            <span class='param-val'>{val}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

with col_out:
    # ── Compute metrics at current θ ──────────────────────────────────────
    y_pred = (y_cal >= theta).astype(int)
    cm     = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    roi          = tp * (Vc - Cr) - fp * Cr - fn * Vc
    roi_nothing  = -n_pos * Vc
    roi_base     = None
    precision    = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_val   = tp / (tp + fn) if (tp + fn) > 0 else 0
    flagged_pct  = (tp + fp) / n * 100
    capacity_ok  = flagged_pct <= capacity

    # baseline θ=0.50
    y_base  = (y_cal >= 0.50).astype(int)
    cm_base = confusion_matrix(y_true, y_base)
    tn_b, fp_b, fn_b, tp_b = cm_base.ravel()
    roi_base = tp_b * (Vc - Cr) - fp_b * Cr - fn_b * Vc

    savings_vs_nothing = roi - roi_nothing
    savings_vs_base    = roi - roi_base

    # ── KPI row ───────────────────────────────────────────────────────────
    roi_color   = "#2ECC71" if roi > 0 else "#E84855"
    cap_color   = "#2ECC71" if capacity_ok else "#E84855"
    prec_color  = "#2ECC71" if precision >= 0.60 else "#F4C430"

    st.markdown(f"""
    <div class='kpi-row'>
        <div class='kpi green'>
            <div class='kpi-val' style='color:{roi_color}'>${roi:,.0f}</div>
            <div class='kpi-lbl'>Net ROI</div>
            <div class='kpi-sub'>vs nothing: +${savings_vs_nothing:,.0f}</div>
        </div>
        <div class='kpi cyan'>
            <div class='kpi-val' style='color:{prec_color}'>{precision:.1%}</div>
            <div class='kpi-lbl'>Precision</div>
            <div class='kpi-sub'>Recall: {recall_val:.1%}</div>
        </div>
        <div class='kpi yellow'>
            <div class='kpi-val' style='color:{cap_color}'>{flagged_pct:.1f}%</div>
            <div class='kpi-lbl'>Flagged %</div>
            <div class='kpi-sub'>capacity limit: {capacity}% {'✓' if capacity_ok else '⚠ exceeded'}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Profit curve ──────────────────────────────────────────────────────
    thresholds = np.arange(0.05, 0.96, 0.01)
    roi_curve  = []
    cap_curve  = []
    for t in thresholds:
        yp = (y_cal >= t).astype(int)
        c  = confusion_matrix(y_true, yp)
        tn_t, fp_t, fn_t, tp_t = c.ravel()
        roi_curve.append(tp_t * (Vc - Cr) - fp_t * Cr - fn_t * Vc)
        cap_curve.append((tp_t + fp_t) / n * 100)

    fig = go.Figure()

    # Capacity shading
    eligible = [r if c <= capacity else None for r, c in zip(roi_curve, cap_curve)]
    fig.add_trace(go.Scatter(
        x=thresholds, y=eligible,
        fill='tozeroy',
        fillcolor='rgba(46,204,113,0.08)',
        line=dict(color='rgba(0,0,0,0)'),
        name='Within capacity',
        showlegend=True,
    ))

    # Full ROI curve
    fig.add_trace(go.Scatter(
        x=thresholds, y=roi_curve,
        line=dict(color='#E84855', width=2.5),
        name='ROI curve',
    ))

    # Markers
    fig.add_vline(x=theta, line_color='#F4C430', line_width=1.5,
                  line_dash='dash', annotation_text=f"θ={theta}",
                  annotation_font_color='#F4C430')
    fig.add_vline(x=0.50,  line_color='#6B7A99', line_width=1,
                  line_dash='dot',  annotation_text="θ=0.50",
                  annotation_font_color='#6B7A99')
    fig.add_hline(y=0, line_color='white', line_width=0.8, opacity=0.3)

    # Current θ ROI dot
    fig.add_trace(go.Scatter(
        x=[theta], y=[roi],
        mode='markers',
        marker=dict(color='#F4C430', size=10, symbol='circle'),
        name=f'Current (${roi:,.0f})',
        showlegend=True,
    ))

    fig.update_layout(
        paper_bgcolor='#111827',
        plot_bgcolor='#111827',
        font=dict(family='IBM Plex Mono', color='#E8EDF5', size=11),
        xaxis=dict(title='Threshold (θ)', gridcolor='#1E2D45',
                   zeroline=False, color='#6B7A99'),
        yaxis=dict(title='Net ROI ($)', gridcolor='#1E2D45',
                   zeroline=False, color='#6B7A99'),
        legend=dict(bgcolor='#1a2236', bordercolor='#1E2D45',
                    borderwidth=1, font=dict(size=11)),
        margin=dict(l=10, r=10, t=30, b=10),
        height=300,
        title=dict(text='Profit Curve — X_test (705 customers)',
                   font=dict(size=12, color='#6B7A99')),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Confusion matrix in $ ──────────────────────────────────────────────
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown(f"<div class='section-title'>Confusion Matrix in $ — θ={theta}</div>",
                unsafe_allow_html=True)

    cm_col1, cm_col2 = st.columns(2, gap="medium")

    cm_data = [
        ("True Negative",  "No action needed",    tn, "$0",
         f"= {tn} clientes × $0",
         "rgba(46,204,113,0.08)", "#2ECC71"),
        ("False Positive", "Wasted intervention", fp, f"−${fp*Cr:,.0f}",
         f"= {fp} clientes × ${Cr} (Cr)",
         "rgba(232,72,85,0.08)", "#E84855"),
        ("False Negative", "Missed churner",      fn, f"−${fn*Vc:,.0f}",
         f"= {fn} clientes × ${Vc} (Vc)",
         "rgba(232,72,85,0.12)", "#E84855"),
        ("True Positive",  "Retained customer",   tp, f"+${tp*(Vc-Cr):,.0f}",
         f"= {tp} clientes × (${Vc}−${Cr})",
         "rgba(46,204,113,0.12)", "#2ECC71"),
    ]
    for (label, desc, count, dollar, formula, bg, color), col in zip(cm_data, [cm_col1, cm_col2, cm_col1, cm_col2]):
        with col:
            st.markdown(f"""
            <div style='background:{bg};border:1px solid {color}33;border-radius:8px;
                        padding:14px 18px;margin-bottom:10px;'>
                <div style='font-size:10px;letter-spacing:2px;text-transform:uppercase;
                            color:{color};font-family:IBM Plex Mono,monospace;margin-bottom:4px;'>
                    {label}</div>
                <div style='font-size:22px;font-weight:700;font-family:IBM Plex Mono,monospace;
                            color:{color};'>{dollar}</div>
                <div style='font-size:11px;color:#8A9BB8;margin-top:3px;'>{desc} · n={count}</div>
                <div style='font-size:11px;color:{color};opacity:0.75;margin-top:5px;
                            font-family:IBM Plex Mono,monospace;'>{formula}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # ── 3 Scenarios comparison ─────────────────────────────────────────────
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Scenario Comparison</div>",
                unsafe_allow_html=True)

    scenarios = [
        ("Do Nothing (θ=1.0)", roi_nothing, "#E84855"),
        ("Baseline (θ=0.50)",  roi_base,    "#6B7A99"),
        (f"Current (θ={theta})", roi,        "#2ECC71" if roi > roi_base else "#F4C430"),
    ]
    fig2 = go.Figure()
    for name, val, color in scenarios:
        fig2.add_trace(go.Bar(
            x=[name], y=[val],
            marker_color=color,
            marker_opacity=0.85,
            text=[f"${val:,.0f}"],
            textposition='outside',
            textfont=dict(family='IBM Plex Mono', size=12, color='#E8EDF5'),
            name=name,
        ))
    fig2.add_hline(y=0, line_color='white', line_width=0.8, opacity=0.4)
    fig2.update_layout(
        paper_bgcolor='#111827', plot_bgcolor='#111827',
        font=dict(family='IBM Plex Mono', color='#E8EDF5', size=11),
        xaxis=dict(gridcolor='#1E2D45', zeroline=False),
        yaxis=dict(title='Net ROI ($)', gridcolor='#1E2D45', zeroline=False, color='#6B7A99'),
        showlegend=False,
        margin=dict(l=10, r=10, t=20, b=10),
        height=260,
        bargap=0.35,
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Top-N flagged customers ────────────────────────────────────────────
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    flagged_n = tp + fp
    st.markdown(f"<div class='section-title'>Top Flagged Customers — {flagged_n} at θ={theta}</div>",
                unsafe_allow_html=True)

    df_flagged = df_test.copy()
    df_flagged['flagged'] = (df_flagged['y_cal'] >= theta).astype(int)
    df_flagged = df_flagged[df_flagged['flagged'] == 1].sort_values('y_cal', ascending=False)

    display = df_flagged[['customer_id', 'y_cal', 'y_true', 'risk_tier']].copy()
    display.columns = ['Customer ID', 'P(churn)', 'True Label', 'Risk Tier']
    display['P(churn)'] = display['P(churn)'].round(3)
    display['True Label'] = display['True Label'].map({1: '🔴 Churned', 0: '🟢 Retained'})

    st.dataframe(
        display.head(20),
        use_container_width=True,
        hide_index=True,
        column_config={
            'P(churn)': st.column_config.ProgressColumn(
                'P(churn)', min_value=0, max_value=1, format="%.3f"),
        }
    )
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("""
<hr style='border-color:#1E2D45;margin-top:16px;'>
<div style='text-align:center;font-size:11px;color:#6B7A99;
            font-family:IBM Plex Mono,monospace;padding-bottom:14px;'>
    ROI Simulator · X_test (705 customers) · Vc and Cr adjustable
</div>
""", unsafe_allow_html=True)
