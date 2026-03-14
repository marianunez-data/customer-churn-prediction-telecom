"""
Page 1 — Model Audit
Model card, calibration, SHAP global importance, dependence plots.
"""

import streamlit as st
import json
from pathlib import Path

st.set_page_config(page_title="Model Audit — Churn Intelligence",
                   page_icon="🔬", layout="wide")

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
.page-header{background:linear-gradient(135deg,#0D1321,#111827);
    border:1px solid var(--border);border-left:4px solid var(--cyan);
    border-radius:10px;padding:26px 34px;margin-bottom:26px;}
.section-card{background:var(--bg-card);border:1px solid var(--border);
    border-radius:10px;padding:24px 28px;margin-bottom:16px;}
.section-title{font-size:10px;letter-spacing:2px;text-transform:uppercase;
    color:var(--cyan);font-family:'IBM Plex Mono',monospace;margin-bottom:14px;}
.ms{display:grid;grid-template-columns:repeat(6,1fr);gap:10px;}
.ms-item{background:var(--bg-card2);border:1px solid var(--border);
    border-radius:8px;padding:13px 15px;text-align:center;}
.ms-val{font-size:20px;font-weight:700;font-family:'IBM Plex Mono',monospace;}
.ms-lbl{font-size:10px;color:var(--muted);letter-spacing:1px;margin-top:3px;}
.ms-sub{font-size:10px;color:var(--muted);margin-top:1px;}
.audit-table{width:100%;border-collapse:collapse;font-size:13px;}
.audit-table th{font-family:'IBM Plex Mono',monospace;font-size:11px;letter-spacing:2px;
    text-transform:uppercase;color:var(--muted);padding:7px 12px;
    text-align:left;border-bottom:1px solid var(--border);}
.audit-table td{padding:9px 12px;border-bottom:1px solid rgba(30,45,69,0.4);color:var(--text);}
.audit-table tr:last-child td{border-bottom:none;}
.audit-table tr:hover td{background:rgba(0,212,255,0.03);}
.insight{padding:13px 17px;background:var(--bg-card2);border-radius:8px;
    font-size:13px;color:var(--muted);line-height:1.7;margin-top:12px;}
.insight strong{color:var(--text);}
code.hl{color:var(--cyan);background:rgba(0,212,255,0.1);
    padding:1px 5px;border-radius:3px;font-family:'IBM Plex Mono',monospace;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

PROJECT_ROOT = Path(__file__).parent.parent.parent

@st.cache_data
def load_data():
    meta = json.loads((PROJECT_ROOT / 'reports' / 'champion_metadata.json').read_text())
    return meta

meta = load_data()

# ── Header ────────────────────────────────────────────────────────────────
st.markdown("""
<div class='page-header'>
    <div style='font-size:10px;letter-spacing:3px;text-transform:uppercase;
                color:#00D4FF;font-family:IBM Plex Mono,monospace;margin-bottom:8px;'>🔬 Model Audit</div>
    <div style='font-size:24px;font-weight:700;margin-bottom:6px;'>Full Diagnostic Report</div>
    <div style='color:#6B7A99;font-size:13px;'>
        Training integrity · Calibration analysis · SHAP global interpretability
    </div>
</div>
""", unsafe_allow_html=True)

# ═══ 1. Performance strip ═════════════════════════════════════════════════
st.markdown("<div class='section-card'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>Performance Across Splits — Definitive Numbers</div>",
            unsafe_allow_html=True)
st.markdown(f"""
<div class='ms'>
    <div class='ms-item'><div class='ms-val' style='color:#00D4FF'>{meta.get('cv_auc_mean',0.9075):.4f}</div>
        <div class='ms-lbl'>CV AUC</div><div class='ms-sub'>±{meta.get('cv_auc_std',0.0057):.4f} · 5-fold</div></div>
    <div class='ms-item'><div class='ms-val' style='color:#F4C430'>{meta.get('val_auc',0.9161):.4f}</div>
        <div class='ms-lbl'>Val AUC</div><div class='ms-sub'>Platt calibrated</div></div>
    <div class='ms-item'><div class='ms-val' style='color:#2ECC71'>{meta.get('test_auc',0.9078):.4f}</div>
        <div class='ms-lbl'>Test AUC</div><div class='ms-sub'>Definitive · honest</div></div>
    <div class='ms-item'><div class='ms-val' style='color:#00D4FF'>{meta.get('test_brier',0.1013):.4f}</div>
        <div class='ms-lbl'>Brier</div><div class='ms-sub'>Test · calibrated</div></div>
    <div class='ms-item'><div class='ms-val' style='color:#2ECC71'>76.0%</div>
        <div class='ms-lbl'>Precision</div><div class='ms-sub'>θ=0.41 · X_test</div></div>
    <div class='ms-item'><div class='ms-val' style='color:#F4C430'>69.5%</div>
        <div class='ms-lbl'>Recall</div><div class='ms-sub'>θ=0.41 · X_test</div></div>
</div>
""", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# ═══ 2. Model Card + Pipeline table ══════════════════════════════════════
c1, c2 = st.columns(2, gap="large")

with c1:
    params = meta.get('best_params', {})
    rows = [
        ('Algorithm',    'LGBMClassifier'),
        ('Tuning',       'FLAML AutoML · 180s · seed=42'),
        ('n_estimators', params.get('n_estimators', 799)),
        ('num_leaves',   params.get('num_leaves', 17)),
        ('learning_rate',f"{params.get('learning_rate',0.069):.4f}"),
        ('Imbalance',    'is_unbalance=True'),
        ('Calibration',  'Platt Scaling (X_val)'),
        ('Features',     '18 (15 categorical + 3 numeric)'),
        ('Train rows',   f"{meta.get('train_rows',5282):,} (75%)"),
        ('Val rows',     f"{meta.get('val_rows',1056):,} (15%)"),
        ('Test rows',    f"{meta.get('test_rows',705):,} (10%)"),
        ('θ_optimal',    meta.get('theta_optimal', 0.41)),
        ('Model size',   f"{meta.get('model_size_kb',1477):.0f} KB"),
    ]
    rows_html = ''.join(
        f"<tr><td style='color:#6B7A99;padding:6px 12px;width:45%'>{k}</td>"
        f"<td style='font-family:IBM Plex Mono,monospace;padding:6px 12px;'>{v}</td></tr>"
        for k,v in rows
    )
    st.markdown(f"""
    <div class='section-card'>
        <div class='section-title'>Model Card</div>
        <table class='audit-table'>{rows_html}</table>
    </div>
    """, unsafe_allow_html=True)

with c2:
    pipe_rows = [
        ("FLAML search",         "X_train fit · X_val eval", "Hyperparameter opt.",   "✓ Clean"),
        ("Champion training",    "X_train only (5,282)",     "Model learning",         "✓ Clean"),
        ("CV stability",         "X_train 5-fold",           "Generalization check",   "✓ Clean"),
        ("Platt calibration",    "X_val — unseen",           "Prob. calibration",      "✓ Honest"),
        ("OOF sweep",            "X_train 5-fold + Platt",   "θ_optimal search",       "✓ Honest"),
        ("Test eval θ=0.41",     "X_test — first & only",    "Final honest ROI",       "✓ Final"),
    ]
    table = """<table class='audit-table'>
    <thead><tr><th>Stage</th><th>Data</th><th>Purpose</th><th>Status</th></tr></thead><tbody>"""
    for stage, data, purpose, status in pipe_rows:
        color = "#2ECC71" if "Clean" in status or "Honest" in status else "#00D4FF"
        table += f"""<tr>
            <td>{stage}</td>
            <td style='font-family:IBM Plex Mono,monospace;color:#6B7A99;font-size:11px'>{data}</td>
            <td style='color:#6B7A99'>{purpose}</td>
            <td style='color:{color};font-family:IBM Plex Mono,monospace;font-size:11px'>{status}</td>
        </tr>"""
    table += "</tbody></table>"
    st.markdown(f"""
    <div class='section-card'>
        <div class='section-title'>Pipeline Integrity</div>
        {table}
    </div>
    """, unsafe_allow_html=True)

# ═══ 3. Calibration ═══════════════════════════════════════════════════════
st.markdown("<div class='section-card'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>Calibration — Platt vs Isotonic vs Raw</div>",
            unsafe_allow_html=True)

ca, cb = st.columns([2, 3], gap="large")
with ca:
    brier_raw  = meta.get('brier_raw_val',  0.1040)
    brier_plat = meta.get('brier_platt_val',0.0993)
    brier_iso  = meta.get('brier_isotonic_val', 0.0927)
    cal_table = f"""<table class='audit-table'>
    <thead><tr><th>Method</th><th>Brier</th><th>Δ vs Raw</th><th>Decision</th></tr></thead>
    <tbody>
    <tr><td>Raw (uncalibrated)</td>
        <td style='font-family:IBM Plex Mono,monospace'>{brier_raw}</td>
        <td style='color:#6B7A99'>—</td><td style='color:#6B7A99'>baseline</td></tr>
    <tr><td>Platt Scaling</td>
        <td style='font-family:IBM Plex Mono,monospace;color:#2ECC71'>{brier_plat}</td>
        <td style='color:#2ECC71'>−{round(brier_raw-brier_plat,4)}</td>
        <td style='color:#2ECC71;font-family:IBM Plex Mono,monospace'>✓ Deployed</td></tr>
    <tr><td>Isotonic Regression</td>
        <td style='font-family:IBM Plex Mono,monospace'>{brier_iso}</td>
        <td style='color:#6B7A99'>−{round(brier_raw-brier_iso,4)}</td>
        <td style='color:#6B7A99;font-size:11px'>overfits small N</td></tr>
    </tbody></table>"""
    st.markdown(cal_table, unsafe_allow_html=True)
    st.markdown("""
    <div class='insight'>
        <strong>Why Platt over Isotonic?</strong><br>
        Isotonic achieves lower Brier on X_val (280 positive samples) but is prone to
        overfitting on small calibration sets. Platt provides monotone, generalizable
        calibration — the conservative, production-safe choice.
    </div>
    """, unsafe_allow_html=True)

with cb:
    img = PROJECT_ROOT / 'reports' / 'champion_calibrated_diagnostics.png'
    if img.exists():
        st.image(str(img), use_container_width=True)
    else:
        st.info("champion_calibrated_diagnostics.png not found in reports/")

st.markdown("</div>", unsafe_allow_html=True)

# ═══ 4. Test set evaluation ════════════════════════════════════════════════
st.markdown("<div class='section-card'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>Official Test Set Evaluation — X_test First & Only Use</div>",
            unsafe_allow_html=True)
img = PROJECT_ROOT / 'reports' / 'champion_test_evaluation.png'
if img.exists():
    st.image(str(img), use_container_width=True)
else:
    st.info("champion_test_evaluation.png not found in reports/")
st.markdown("</div>", unsafe_allow_html=True)

# ═══ 5. SHAP global ════════════════════════════════════════════════════════
st.markdown("<div class='section-card'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>SHAP Global Feature Importance — X_test (705 rows)</div>",
            unsafe_allow_html=True)

sa, sb = st.columns(2, gap="large")
with sa:
    img = PROJECT_ROOT / 'reports' / 'shap_global_bar.png'
    if img.exists():
        st.image(str(img), use_container_width=True)
with sb:
    img = PROJECT_ROOT / 'reports' / 'shap_global_beeswarm.png'
    if img.exists():
        st.image(str(img), use_container_width=True)

st.markdown("""
<div class='insight' style='border-left:3px solid #00D4FF;'>
    <strong>Key insight:</strong>
    <code class='hl'>tenure_days</code> dominates with mean |SHAP|=1.73 —
    new customers churn at the highest rate.
    <code class='hl'>type</code> (contract) is the second strongest signal —
    month-to-month → high risk, two-year → retention anchor.
    Demographics (gender, dependents) contribute least —
    churn is driven by <strong>relationship depth</strong>, not who the customer is.
</div>
""", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# ═══ 6. SHAP dependence ════════════════════════════════════════════════════
st.markdown("<div class='section-card'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>SHAP Dependence Plots — Top 3 Features</div>",
            unsafe_allow_html=True)
img = PROJECT_ROOT / 'reports' / 'shap_dependence_top3.png'
if img.exists():
    st.image(str(img), use_container_width=True)
else:
    st.info("shap_dependence_top3.png not found in reports/")

st.markdown("""
<div class='insight' style='border-left:3px solid #F4C430;'>
    <strong>Three features, one story:</strong>
    New customers (low tenure, low total_charges) on month-to-month contracts represent
    the maximum risk profile. <code class='hl'>tenure_days</code> and
    <code class='hl'>total_charges</code> are correlated proxies — both capture
    customer longevity from different angles.
</div>
""", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("""
<hr style='border-color:#1E2D45;margin-top:16px;'>
<div style='text-align:center;font-size:11px;color:#6B7A99;
            font-family:IBM Plex Mono,monospace;padding-bottom:14px;'>
    Model Audit · LightGBM + Platt · AUC 0.9078 · Brier 0.1013
</div>
""", unsafe_allow_html=True)
