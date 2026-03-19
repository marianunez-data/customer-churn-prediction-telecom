"""
Page 3 — Live Scoring
Modo A: buscar customer por ID → perfil + SHAP waterfall
Modo B: formulario manual → predicción en vivo con el modelo
"""

import streamlit as st
import json
import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(page_title="Live Scoring — Churn Intelligence",
                   page_icon="🎯", layout="wide")

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
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div input {
    background:#1a2236!important;color:#E8EDF5!important;
    border-color:#1E2D45!important;font-family:'IBM Plex Mono',monospace!important;}
.stSlider label p,.stSelectbox label p,.stNumberInput label p{
    color:var(--text)!important;font-size:13px!important;}
.page-header{background:linear-gradient(135deg,#0D1321,#111827);
    border:1px solid var(--border);border-left:4px solid var(--red);
    border-radius:10px;padding:26px 34px;margin-bottom:24px;}
.section-card{background:var(--bg-card);border:1px solid var(--border);
    border-radius:10px;padding:22px 26px;margin-bottom:14px;}
.section-title{font-size:10px;letter-spacing:2px;text-transform:uppercase;
    color:var(--cyan);font-family:'IBM Plex Mono',monospace;margin-bottom:14px;}
.profile-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin-bottom:16px;}
.prof-item{background:var(--bg-card2);border:1px solid var(--border);
    border-radius:8px;padding:12px 15px;}
.prof-lbl{font-size:10px;letter-spacing:1.5px;text-transform:uppercase;
    color:var(--muted);font-family:'IBM Plex Mono',monospace;margin-bottom:5px;}
.prof-val{font-size:14px;font-weight:600;color:var(--text);}
.risk-badge{display:inline-block;padding:6px 18px;border-radius:20px;
    font-family:'IBM Plex Mono',monospace;font-size:13px;font-weight:600;
    letter-spacing:1px;text-transform:uppercase;}
.risk-high{background:rgba(232,72,85,0.15);color:#E84855;border:1px solid rgba(232,72,85,0.4);}
.risk-med {background:rgba(244,196,48,0.12);color:#F4C430;border:1px solid rgba(244,196,48,0.3);}
.risk-low {background:rgba(46,204,113,0.12);color:#2ECC71;border:1px solid rgba(46,204,113,0.3);}
.result-hero{background:var(--bg-card2);border:1px solid var(--border);border-radius:12px;
    padding:28px 36px;margin-bottom:16px;display:flex;align-items:center;gap:32px;}
.result-prob{font-size:52px;font-weight:700;font-family:'IBM Plex Mono',monospace;line-height:1;}
.result-meta{flex:1;}
.result-label{font-size:11px;letter-spacing:3px;text-transform:uppercase;
    color:var(--muted);font-family:'IBM Plex Mono',monospace;margin-bottom:6px;}
.result-desc{font-size:14px;color:var(--muted);line-height:1.6;margin-top:8px;}
.divider{border:none;border-top:1px solid var(--border);margin:20px 0;}
.info-note{background:rgba(0,212,255,0.06);border:1px solid rgba(0,212,255,0.18);
    border-radius:8px;padding:11px 15px;font-size:12px;color:var(--muted);
    font-family:'IBM Plex Mono',monospace;margin-bottom:14px;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

PROJECT_ROOT = Path(__file__).parent.parent.parent

# ── Load artifacts ─────────────────────────────────────────────────────────
@st.cache_resource
def load_artifact():
    return joblib.load(PROJECT_ROOT / 'models' / 'champion_calibrated.pkl')

@st.cache_data
def load_shap_precomputed():
    vals  = np.load(PROJECT_ROOT / 'reports' / 'shap_values.npy')
    base  = np.load(PROJECT_ROOT / 'reports' / 'shap_base_values.npy')
    data  = np.load(PROJECT_ROOT / 'reports' / 'shap_data.npy')
    names = json.loads((PROJECT_ROOT / 'reports' / 'shap_feature_names.json').read_text())
    return vals, base, data, names

@st.cache_data
def load_test_predictions():
    return pd.read_csv(PROJECT_ROOT / 'reports' / 'final_results_test.csv')

@st.cache_data
def load_original_features():
    """Load original (pre-encoded) feature values for profile display."""
    p = PROJECT_ROOT / 'data' / 'processed' / 'demo_customers.csv'
    if p.exists():
        return pd.read_csv(p, index_col=0)
    return None

@st.cache_data
def build_shap_index(test_preds_df):
    return {row['customer_id']: idx
            for idx, row in test_preds_df.reset_index(drop=True).iterrows()}

artifact   = load_artifact()
shap_vals, shap_base, shap_data, feat_names = load_shap_precomputed()
test_preds     = load_test_predictions()
shap_index     = build_shap_index(test_preds)
original_feats = load_original_features()

pipe      = artifact['base_pipeline']
platt     = artifact['calibrator']
threshold = artifact['deployed_threshold']
cat_cols  = artifact['cat_cols']
num_cols  = artifact['num_cols']
REQUIRED_COLS = cat_cols + num_cols

# Print exact categories known by encoder (for debugging)
try:
    enc = pipe.named_steps['preprocessor'].transformers_[0][1]
    KNOWN_CATS = {col: list(cats) for col, cats in zip(cat_cols, enc.categories_)}
except Exception:
    KNOWN_CATS = {}

# ── Helpers ────────────────────────────────────────────────────────────────
def infer_single(row_dict: dict) -> dict:
    df = pd.DataFrame([row_dict])
    # sklearn 1.8 + Python 3.14 are strict — no mixed types allowed
    # Cast every cat col to clean str, every num col to float64
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').astype('float64')
    # Force object dtype on cat cols to avoid pandas inferring int/float
    df[cat_cols] = df[cat_cols].astype(object)
    df = df[REQUIRED_COLS]
    try:
        raw = pipe.predict_proba(df)[:, 1]
    except Exception as e:
        st.error(f"Pipeline error: {e}")
        return {'p_churn': 0.0, 'flagged': 0, 'risk_tier': 'Low'}
    cal  = platt.predict_proba(raw.reshape(-1, 1))[:, 1][0]
    tier = 'High' if cal >= 0.60 else ('Medium' if cal >= 0.30 else 'Low')
    return {'p_churn': cal, 'flagged': int(cal >= threshold), 'risk_tier': tier}


def compute_shap_single(row_dict: dict):
    import shap
    df = pd.DataFrame([row_dict])
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').astype('float64')
    df[cat_cols] = df[cat_cols].astype(object)
    preprocessor = pipe.named_steps['preprocessor']
    classifier   = pipe.named_steps['classifier']
    X_t = preprocessor.transform(df[REQUIRED_COLS])
    explainer = shap.TreeExplainer(classifier,
                                   feature_perturbation='tree_path_dependent')
    sv = explainer(X_t)
    sv.feature_names = feat_names
    return sv


def waterfall_plotly(sv_row, bv, feat_names, feat_vals, title, p_churn):
    idx    = np.argsort(np.abs(sv_row))[::-1][:12]
    sv_s   = sv_row[idx]
    fn_s   = [feat_names[i] for i in idx]
    fv_s   = [feat_vals[i]  for i in idx]
    labels = [f"{n} = {v}" for n, v in zip(fn_s, fv_s)]

    fig = go.Figure(go.Waterfall(
        orientation='h',
        measure=['relative'] * len(sv_s) + ['total'],
        x=list(sv_s) + [float(np.sum(sv_row))],
        y=labels + ['Final Score'],
        connector=dict(line=dict(color='#1E2D45', width=1)),
        increasing=dict(marker_color='#E84855'),
        decreasing=dict(marker_color='#00D4FF'),
        totals=dict(marker_color='#F4C430'),
        text=[f"{v:+.3f}" for v in list(sv_s) + [float(np.sum(sv_row))]],
        textposition='outside',
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(size=13, color='#E8EDF5',
                                         family='IBM Plex Mono')),
        paper_bgcolor='#111827', plot_bgcolor='#111827',
        font=dict(family='IBM Plex Mono', color='#E8EDF5', size=11),
        xaxis=dict(title='SHAP value (impact on churn probability)',
                   gridcolor='#1E2D45', zeroline=True,
                   zerolinecolor='#6B7A99', color='#6B7A99'),
        yaxis=dict(gridcolor='#1E2D45', color='#6B7A99'),
        margin=dict(l=10, r=10, t=46, b=10),
        height=420,
    )
    return fig


def risk_badge(tier):
    cls   = {'High': 'risk-high', 'Medium': 'risk-med', 'Low': 'risk-low'}
    icons = {'High': '🔴', 'Medium': '🟡', 'Low': '🟢'}
    return f"<span class='risk-badge {cls[tier]}'>{icons[tier]} {tier} Risk</span>"


def prob_color(p):
    if p >= 0.60: return '#E84855'
    if p >= 0.30: return '#F4C430'
    return '#2ECC71'


# ── Header ────────────────────────────────────────────────────────────────
st.markdown("""
<div class='page-header'>
    <div style='font-size:10px;letter-spacing:3px;text-transform:uppercase;
                color:#E84855;font-family:IBM Plex Mono,monospace;margin-bottom:8px;'>
        🎯 Live Scoring</div>
    <div style='font-size:24px;font-weight:700;margin-bottom:6px;'>Customer Risk Scoring</div>
    <div style='font-size:13px;color:#8A9BB8;'>
        Search any customer by ID · Build a manual profile · Instant predictions + SHAP explanations
    </div>
</div>
""", unsafe_allow_html=True)

# ── Mode tabs ─────────────────────────────────────────────────────────────
mode = st.radio("", ["🔍  Search by Customer ID", "🧪  Manual Profile Builder"],
                horizontal=True, label_visibility="collapsed")

st.markdown("<hr class='divider'>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# MODE A — Search by Customer ID
# ══════════════════════════════════════════════════════════════════════════
if mode == "🔍  Search by Customer ID":

    col_search, col_info = st.columns([1, 2], gap="large")

    with col_search:
        st.markdown("<div class='section-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Search Customer</div>",
                    unsafe_allow_html=True)

        total   = len(test_preds)
        flagged = int(test_preds['y_pred_optimal'].sum())
        high_r  = int((test_preds['risk_tier'] == 'High').sum())

        st.markdown(f"""
        <div style='display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-bottom:16px;'>
            <div style='background:#1a2236;border:1px solid #1E2D45;border-radius:8px;
                        padding:10px;text-align:center;'>
                <div style='font-size:20px;font-weight:700;font-family:IBM Plex Mono,monospace;
                            color:#00D4FF;'>{total}</div>
                <div style='font-size:10px;color:#8A9BB8;margin-top:2px;'>Total</div>
            </div>
            <div style='background:#1a2236;border:1px solid #1E2D45;border-radius:8px;
                        padding:10px;text-align:center;'>
                <div style='font-size:20px;font-weight:700;font-family:IBM Plex Mono,monospace;
                            color:#E84855;'>{flagged}</div>
                <div style='font-size:10px;color:#8A9BB8;margin-top:2px;'>Flagged</div>
            </div>
            <div style='background:#1a2236;border:1px solid #1E2D45;border-radius:8px;
                        padding:10px;text-align:center;'>
                <div style='font-size:20px;font-weight:700;font-family:IBM Plex Mono,monospace;
                            color:#F4C430;'>{high_r}</div>
                <div style='font-size:10px;color:#8A9BB8;margin-top:2px;'>High Risk</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        all_ids     = sorted(test_preds['customer_id'].tolist())
        selected_id = st.selectbox("Customer ID", all_ids, index=0)

        st.markdown("<hr class='divider' style='margin:14px 0;'>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:10px;letter-spacing:2px;text-transform:uppercase;"
                    "color:#00D4FF;font-family:IBM Plex Mono,monospace;margin-bottom:10px;'>"
                    "Top 10 Highest Risk</div>", unsafe_allow_html=True)

        top10 = (test_preds.nlargest(10, 'y_cal')
                 [['customer_id', 'y_cal', 'risk_tier']]
                 .reset_index(drop=True))

        for _, row in top10.iterrows():
            is_sel = row['customer_id'] == selected_id
            bg     = "rgba(0,212,255,0.08)" if is_sel else "transparent"
            border = "#00D4FF"              if is_sel else "#1E2D45"
            c      = prob_color(row['y_cal'])
            st.markdown(f"""
            <div style='background:{bg};border:1px solid {border};border-radius:6px;
                        padding:7px 12px;margin-bottom:5px;display:flex;
                        justify-content:space-between;align-items:center;'>
                <span style='font-family:IBM Plex Mono,monospace;font-size:12px;'>
                    #{row['customer_id']}</span>
                <span style='font-family:IBM Plex Mono,monospace;font-size:13px;
                             font-weight:700;color:{c};'>{row['y_cal']:.3f}</span>
            </div>
            """, unsafe_allow_html=True)


    with col_info:
        row  = test_preds[test_preds['customer_id'] == selected_id].iloc[0]
        p    = row['y_cal']
        tier = row['risk_tier']
        pc   = prob_color(p)
        flag_str  = "🚨 Flagged for intervention" if row['y_pred_optimal'] == 1 \
                    else "✓ Not flagged"
        true_label = "🔴 Churned" if row['y_true'] == 1 else "🟢 Retained"

        st.markdown(f"""
        <div class='result-hero' style='border-left:4px solid {pc};'>
            <div>
                <div class='result-label'>Churn Probability</div>
                <div class='result-prob' style='color:{pc};'>{p:.1%}</div>
            </div>
            <div class='result-meta'>
                <div style='margin-bottom:8px;'>{risk_badge(tier)}</div>
                <div style='font-size:13px;color:#8A9BB8;margin-bottom:4px;'>
                    {flag_str} · θ={threshold}</div>
                <div style='font-size:13px;color:#8A9BB8;'>
                    Ground truth: {true_label}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if selected_id in shap_index:
            shap_idx = shap_index[selected_id]
            sv_row   = shap_vals[shap_idx]
            bv_row   = float(shap_base[shap_idx]) if shap_base.ndim > 0 else float(shap_base)
            fv_row   = shap_data[shap_idx]

            with st.container(border=True):
                st.markdown("<div class='section-title'>Customer Profile</div>",
                            unsafe_allow_html=True)
                # Use original (pre-encoded) values if available
                if original_feats is not None and selected_id in original_feats.index:
                    orig_row = original_feats.loc[selected_id]
                    profile_items = [(name, orig_row.get(name, fv_row[i]))
                                     for i, name in enumerate(feat_names)]
                else:
                    profile_items = [(name, fv_row[i])
                                     for i, name in enumerate(feat_names)]

                profile_html = "<div class='profile-grid'>"
                for name, val in profile_items:
                    display_val = f'{val:.2f}' if isinstance(val, float) and not isinstance(val, int) else str(val)
                    profile_html += (f"<div class='prof-item'>"
                                     f"<div class='prof-lbl'>{name}</div>"
                                     f"<div class='prof-val'>{display_val}</div></div>")
                profile_html += "</div>"
                st.markdown(profile_html, unsafe_allow_html=True)

            with st.container(border=True):
                st.markdown("<div class='section-title'>SHAP Explanation — Why this score?</div>",
                            unsafe_allow_html=True)
                fig = waterfall_plotly(sv_row, bv_row, feat_names, fv_row,
                                       f"Customer {selected_id} — P(churn)={p:.3f}", p)
                st.plotly_chart(fig, width="stretch")

                feat_df = (pd.DataFrame({
                    'Feature':     feat_names,
                    'Value':       [f'{v:.2f}' if isinstance(v, float) else str(v)
                                    for v in fv_row],
                    'SHAP Impact': sv_row.round(4),
                    'Direction':   ['🔴 Increases risk' if v > 0 else '🔵 Reduces risk'
                                    for v in sv_row],
                }).sort_values('SHAP Impact', key=abs, ascending=False).head(8))
                st.dataframe(feat_df, width="stretch", hide_index=True)


# ══════════════════════════════════════════════════════════════════════════
# MODE B — Manual Profile Builder
# ══════════════════════════════════════════════════════════════════════════
else:
    st.markdown("""
    <div class='info-note'>
        ℹ Build a hypothetical customer profile and get an instant churn prediction.
        SHAP values computed with tree_path_dependent — valid and interpretable.
    </div>
    """, unsafe_allow_html=True)

    col_form, col_result = st.columns([1, 1], gap="large")

    with col_form:
        with st.container(border=True):
            st.markdown("<div class='section-title'>Contract & Services</div>",
                        unsafe_allow_html=True)
            # Use exact categories from the trained encoder when available
            def opts(col, fallback):
                return KNOWN_CATS.get(col, fallback)

            f_type     = st.selectbox("Contract Type",
                                      opts('type', ['Month-to-month', 'One year', 'Two year']),
                                      help="Month-to-month = highest churn risk")
            f_internet = st.selectbox("Internet Service",
                                      opts('internet_service', ['DSL', 'Fiber optic', 'No']))
            f_security = st.selectbox("Online Security",
                                      opts('online_security', ['No', 'No internet service', 'Yes']))
            f_backup   = st.selectbox("Online Backup",
                                      opts('online_backup', ['No', 'No internet service', 'Yes']))
            f_device   = st.selectbox("Device Protection",
                                      opts('device_protection', ['No', 'No internet service', 'Yes']))
            f_tech     = st.selectbox("Tech Support",
                                      opts('tech_support', ['No', 'No internet service', 'Yes']))
            f_tv       = st.selectbox("Streaming TV",
                                      opts('streaming_tv', ['No', 'No internet service', 'Yes']))
            f_movies   = st.selectbox("Streaming Movies",
                                      opts('streaming_movies', ['No', 'No internet service', 'Yes']))
            f_lines    = st.selectbox("Multiple Lines",
                                      opts('multiple_lines', ['No', 'No phone service', 'Yes']))

        with st.container(border=True):
            st.markdown("<div class='section-title'>Billing & Demographics</div>",
                        unsafe_allow_html=True)
            f_paperless = st.selectbox("Paperless Billing",
                                       opts('paperless_billing', ['No', 'Yes']))
            f_payment   = st.selectbox("Payment Method",
                                       opts('payment_method',
                                            ['Bank transfer (automatic)',
                                             'Credit card (automatic)',
                                             'Electronic check',
                                             'Mailed check']))
            f_gender    = st.selectbox("Gender",         opts('gender', ['Female', 'Male']))
            f_senior    = st.selectbox("Senior Citizen", ['No', 'Yes'])
            f_partner   = st.selectbox("Partner",        opts('partner', ['No', 'Yes']))
            f_depends   = st.selectbox("Dependents",     opts('dependents', ['No', 'Yes']))

        with st.container(border=True):
            st.markdown("<div class='section-title'>Usage & Tenure</div>",
                        unsafe_allow_html=True)
            f_tenure  = st.slider("Tenure (days)",        0,   2555, 180,
                                  help="Low tenure = highest risk zone")
            f_monthly = st.slider("Monthly Charges ($)", 18.0, 120.0, 70.0, step=0.5)
            f_total   = st.slider("Total Charges ($)",    0.0, 9000.0,
                                  float(round(f_tenure * f_monthly / 30, 2)),
                                  step=10.0)

    with col_result:
        profile = {
            'type':              f_type,
            'paperless_billing': f_paperless,
            'payment_method':    f_payment,
            'gender':            f_gender,
            'senior_citizen':    1 if f_senior == 'Yes' else 0,
            'partner':           f_partner,
            'dependents':        f_depends,
            'internet_service':  f_internet,
            'online_security':   f_security,
            'online_backup':     f_backup,
            'device_protection': f_device,
            'tech_support':      f_tech,
            'streaming_tv':      f_tv,
            'streaming_movies':  f_movies,
            'multiple_lines':    f_lines,
            'monthly_charges':   f_monthly,
            'total_charges':     f_total,
            'tenure_days':       f_tenure,
        }

        result = infer_single(profile)
        p    = result['p_churn']
        tier = result['risk_tier']
        pc   = prob_color(p)
        flag = result['flagged']

        flag_str = "🚨 Would be flagged" if flag else "✓ Would NOT be flagged"
        desc_str = ('High risk customer. Immediate outreach recommended.' if tier == 'High'
                    else 'Monitor closely — medium churn signal.'          if tier == 'Medium'
                    else 'Low risk. No intervention needed at this threshold.')

        st.markdown(f"""
        <div class='result-hero' style='border-left:4px solid {pc};'>
            <div>
                <div class='result-label'>Churn Probability</div>
                <div class='result-prob' style='color:{pc};'>{p:.1%}</div>
            </div>
            <div class='result-meta'>
                <div style='margin-bottom:10px;'>{risk_badge(tier)}</div>
                <div style='font-size:13px;color:#8A9BB8;'>{flag_str} · θ={threshold}</div>
                <div class='result-desc'>{desc_str}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Gauge
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(p * 100, 1),
            number={'suffix': '%', 'font': {'size': 36, 'color': pc,
                                             'family': 'IBM Plex Mono'}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': '#6B7A99',
                         'tickfont': {'size': 11, 'family': 'IBM Plex Mono'}},
                'bar':  {'color': pc, 'thickness': 0.25},
                'bgcolor': '#1a2236', 'borderwidth': 0,
                'steps': [
                    {'range': [0,  30], 'color': 'rgba(46,204,113,0.12)'},
                    {'range': [30, 60], 'color': 'rgba(244,196,48,0.12)'},
                    {'range': [60,100], 'color': 'rgba(232,72,85,0.12)'},
                ],
                'threshold': {
                    'line': {'color': '#F4C430', 'width': 2},
                    'thickness': 0.75,
                    'value': threshold * 100,
                },
            },
            title={'text': f'P(Churn) — θ={threshold} decision line',
                   'font': {'size': 12, 'color': '#8A9BB8',
                            'family': 'IBM Plex Mono'}},
        ))
        fig_g.update_layout(
            paper_bgcolor='#111827',
            font=dict(color='#E8EDF5'),
            height=260,
            margin=dict(l=20, r=20, t=40, b=10),
        )
        st.plotly_chart(fig_g, width="stretch")

        # SHAP on demand
        with st.container(border=True):
            st.markdown("<div class='section-title'>SHAP Explanation</div>",
                        unsafe_allow_html=True)

            if st.button("⚡ Explain this prediction", width="stretch"):
                with st.spinner("Computing SHAP values..."):
                    sv_new = compute_shap_single(profile)
                    sv_row = sv_new.values[0]
                    bv_row = float(sv_new.base_values[0])
                    fv_row = sv_new.data[0]

                fig_wf = waterfall_plotly(sv_row, bv_row, feat_names, fv_row,
                                          "Manual Profile — SHAP Waterfall", p)
                st.plotly_chart(fig_wf, width="stretch")

                feat_df = (pd.DataFrame({
                    'Feature':     feat_names,
                    'Value':       [f'{v:.2f}' if isinstance(v, float) else str(v)
                                    for v in fv_row],
                    'SHAP Impact': sv_row.round(4),
                    'Direction':   ['🔴 Increases risk' if v > 0 else '🔵 Reduces risk'
                                    for v in sv_row],
                }).sort_values('SHAP Impact', key=abs, ascending=False).head(8))
                st.dataframe(feat_df, width="stretch", hide_index=True)
            else:
                st.markdown("""
                <div style='text-align:center;padding:32px;color:#8A9BB8;
                            font-family:IBM Plex Mono,monospace;font-size:12px;'>
                    Click above to compute<br>SHAP feature contributions
                </div>
                """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────
st.markdown("""
<hr style='border-color:#1E2D45;margin-top:16px;'>
<div style='text-align:center;font-size:11px;color:#8A9BB8;
            font-family:IBM Plex Mono,monospace;padding-bottom:14px;'>
    Live Scoring · champion_calibrated.pkl · SHAP TreeExplainer · θ=0.41
</div>
""", unsafe_allow_html=True)
