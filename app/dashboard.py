"""
Customer Churn Intelligence System — Interconnect Telecom
Home page: KPIs, project summary, navigation.
"""

import streamlit as st
import json
from pathlib import Path

st.set_page_config(
    page_title="Churn Intelligence — Interconnect",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=DM+Sans:wght@300;400;600;700&display=swap');
:root {
    --bg-deep:#0A0F1E; --bg-card:#111827; --bg-card2:#1a2236;
    --cyan:#00D4FF; --red:#E84855; --green:#2ECC71; --yellow:#F4C430;
    --text:#E8EDF5; --muted:#8A9BB8; --border:#1E2D45;
}
html,body,[data-testid="stAppViewContainer"]{
    background:var(--bg-deep)!important;color:var(--text)!important;
    font-family:'DM Sans',sans-serif;
}
[data-testid="stSidebar"]{background:#0D1321!important;border-right:1px solid var(--border);}
[data-testid="stSidebar"] *{color:var(--text)!important;}
h1,h2,h3,h4{font-family:'DM Sans',sans-serif;color:var(--text)!important;}

.hero{background:linear-gradient(135deg,#0D1321 0%,#111827 50%,#0D1F3C 100%);
    border:1px solid var(--border);border-top:3px solid var(--cyan);
    border-radius:12px;padding:48px 56px;margin-bottom:28px;position:relative;overflow:hidden;}
.hero::before{content:'';position:absolute;top:-60px;right:-60px;width:300px;height:300px;
    background:radial-gradient(circle,rgba(0,212,255,0.06) 0%,transparent 70%);}
.hero-tag{font-family:'IBM Plex Mono',monospace;font-size:11px;letter-spacing:3px;
    color:var(--cyan);text-transform:uppercase;margin-bottom:14px;}
.hero-title{font-size:36px;font-weight:700;line-height:1.15;margin-bottom:12px;color:var(--text);}
.hero-title span{color:var(--cyan);}
.hero-sub{font-size:15px;color:var(--muted);line-height:1.7;max-width:600px;}

.kpi-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-bottom:26px;}
.kpi-card{background:var(--bg-card);border:1px solid var(--border);border-radius:10px;
    padding:20px 24px;position:relative;overflow:hidden;}
.kpi-card::after{content:'';position:absolute;bottom:0;left:0;right:0;height:2px;}
.kpi-card.cyan::after{background:var(--cyan);}
.kpi-card.green::after{background:var(--green);}
.kpi-card.yellow::after{background:var(--yellow);}
.kpi-card.red::after{background:var(--red);}
.kpi-label{font-size:10px;letter-spacing:2px;text-transform:uppercase;
    color:var(--muted);font-family:'IBM Plex Mono',monospace;margin-bottom:8px;}
.kpi-value{font-size:30px;font-weight:700;font-family:'IBM Plex Mono',monospace;
    line-height:1;margin-bottom:5px;}
.kpi-value.cyan{color:var(--cyan);}
.kpi-value.green{color:var(--green);}
.kpi-value.yellow{color:var(--yellow);}
.kpi-value.red{color:var(--red);}
.kpi-sub{font-size:11px;color:var(--muted);}

.section-card{background:var(--bg-card);border:1px solid var(--border);
    border-radius:10px;padding:24px 28px;margin-bottom:16px;}
.section-title{font-size:10px;letter-spacing:2px;text-transform:uppercase;
    color:var(--cyan);font-family:'IBM Plex Mono',monospace;margin-bottom:14px;}

.nav-card{background:var(--bg-card2);border:1px solid var(--border);
    border-radius:10px;padding:20px 22px;margin-bottom:10px;}
.nav-icon{font-size:24px;margin-bottom:8px;}
.nav-title{font-size:14px;font-weight:700;color:var(--text);margin-bottom:4px;}
.nav-desc{font-size:12px;color:var(--muted);line-height:1.5;}

.ms{display:grid;grid-template-columns:repeat(6,1fr);gap:10px;}
.ms-item{background:var(--bg-card2);border:1px solid var(--border);
    border-radius:8px;padding:13px 15px;text-align:center;}
.ms-val{font-size:20px;font-weight:700;font-family:'IBM Plex Mono',monospace;}
.ms-lbl{font-size:10px;color:var(--muted);letter-spacing:1px;margin-top:3px;}
.ms-sub{font-size:10px;color:var(--muted);margin-top:1px;}

.pipe-table{width:100%;border-collapse:collapse;font-size:13px;}
.pipe-table th{font-family:'IBM Plex Mono',monospace;font-size:11px;letter-spacing:2px;
    text-transform:uppercase;color:var(--muted);padding:7px 12px;
    text-align:left;border-bottom:1px solid var(--border);}
.pipe-table td{padding:9px 12px;border-bottom:1px solid rgba(30,45,69,0.5);color:var(--text);}
.pipe-table tr:last-child td{border-bottom:none;}
.tag-green{color:#2ECC71;font-family:'IBM Plex Mono',monospace;font-size:12px;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

PROJECT_ROOT = Path(__file__).parent.parent

@st.cache_data
def load_meta():
    p = PROJECT_ROOT / 'reports' / 'champion_metadata.json'
    return json.loads(p.read_text()) if p.exists() else {}

meta = load_meta()

# ── Hero ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class='hero'>
    <div class='hero-tag'>⚡ ML Retention System · Interconnect Telecom</div>
    <div class='hero-title'>Customer Churn<br><span>Intelligence System</span></div>
    <div class='hero-sub'>
        End-to-end ML pipeline for proactive churn prevention. Calibrated predictions,
        business-validated thresholds, and individual SHAP explanations — production-ready.
    </div>
</div>
""", unsafe_allow_html=True)

# ── KPI strip ─────────────────────────────────────────────────────────────
st.markdown(f"""
<div class='kpi-grid'>
    <div class='kpi-card cyan'>
        <div class='kpi-label'>Test AUC</div>
        <div class='kpi-value cyan'>{meta.get('test_auc', 0.9078)}</div>
        <div class='kpi-sub'>CV: 0.9075 ± 0.0057</div>
    </div>
    <div class='kpi-card green'>
        <div class='kpi-label'>Net ROI — Test Cohort</div>
        <div class='kpi-value green'>+${meta.get('roi_optimal_test', 1617):,.0f}</div>
        <div class='kpi-sub'>vs −$12,903 do-nothing</div>
    </div>
    <div class='kpi-card yellow'>
        <div class='kpi-label'>Optimal Threshold</div>
        <div class='kpi-value yellow'>θ={meta.get('theta_optimal', 0.41)}</div>
        <div class='kpi-sub'>OOF sweep · capacity ≤25%</div>
    </div>
    <div class='kpi-card red'>
        <div class='kpi-label'>Precision @ θ_opt</div>
        <div class='kpi-value red'>76.0%</div>
        <div class='kpi-sub'>3 in 4 contacts are real churners</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Layout ────────────────────────────────────────────────────────────────
col_l, col_r = st.columns([3, 2], gap="large")

with col_l:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Pipeline Integrity — Zero Leakage</div>",
                unsafe_allow_html=True)
    rows = [
        ("FLAML AutoML search",      "X_train fit · X_val eval only",  "Hyperparameter search"),
        ("Champion training",         "X_train only — 5,282 rows",      "Model learning"),
        ("CV stability (5-fold)",     "X_train only",                   "Generalization check"),
        ("Platt calibration",         "X_val — unseen by champion",     "Probability calibration"),
        ("OOF threshold sweep",       "X_train 5-fold + Platt",         "θ_optimal search"),
        ("Official test eval θ=0.41","X_test — first & only use",      "Final honest ROI"),
    ]
    table = "<table class='pipe-table'><thead><tr>"
    for h in ["Stage", "Data Used", "Purpose", "Status"]:
        table += f"<th>{h}</th>"
    table += "</tr></thead><tbody>"
    for stage, data, purpose in rows:
        table += f"""<tr>
            <td>{stage}</td>
            <td style='font-family:IBM Plex Mono,monospace;color:#6B7A99;'>{data}</td>
            <td style='color:#6B7A99;'>{purpose}</td>
            <td class='tag-green'>✓ Clean</td>
        </tr>"""
    table += "</tbody></table>"
    st.markdown(table, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Performance Across Splits</div>",
                unsafe_allow_html=True)
    st.markdown(f"""
    <div class='ms'>
        <div class='ms-item'><div class='ms-val' style='color:#00D4FF'>{meta.get('cv_auc_mean',0.9075):.4f}</div>
            <div class='ms-lbl'>CV AUC</div><div class='ms-sub'>±{meta.get('cv_auc_std',0.0057):.4f}</div></div>
        <div class='ms-item'><div class='ms-val' style='color:#F4C430'>{meta.get('val_auc',0.9161):.4f}</div>
            <div class='ms-lbl'>Val AUC</div><div class='ms-sub'>Platt calibrated</div></div>
        <div class='ms-item'><div class='ms-val' style='color:#2ECC71'>{meta.get('test_auc',0.9078):.4f}</div>
            <div class='ms-lbl'>Test AUC</div><div class='ms-sub'>Definitive</div></div>
        <div class='ms-item'><div class='ms-val' style='color:#00D4FF'>{meta.get('test_brier',0.1013):.4f}</div>
            <div class='ms-lbl'>Brier</div><div class='ms-sub'>Calibrated</div></div>
        <div class='ms-item'><div class='ms-val' style='color:#2ECC71'>76.0%</div>
            <div class='ms-lbl'>Precision</div><div class='ms-sub'>θ=0.41</div></div>
        <div class='ms-item'><div class='ms-val' style='color:#F4C430'>69.5%</div>
            <div class='ms-lbl'>Recall</div><div class='ms-sub'>θ=0.41</div></div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col_r:
    st.markdown("<div class='section-card'>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Explore the System</div>",
                unsafe_allow_html=True)
    st.markdown("""
    <div class='nav-card'>
        <div class='nav-icon'>🔬</div>
        <div class='nav-title'>Model Audit</div>
        <div class='nav-desc'>Model card, calibration diagnostics,
        SHAP global importance and dependence plots.</div>
    </div>
    <div class='nav-card'>
        <div class='nav-icon'>💰</div>
        <div class='nav-title'>ROI Simulator</div>
        <div class='nav-desc'>Interactive threshold, cost and capacity sliders.
        Real-time profit curve and confusion matrix in dollars.</div>
    </div>
    <div class='nav-card'>
        <div class='nav-icon'>🎯</div>
        <div class='nav-title'>Live Scoring</div>
        <div class='nav-desc'>Score new customers, rank by churn risk,
        SHAP waterfall on-demand, export to Excel.</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    params = meta.get('best_params', {})
    rows_mc = [
        ('Algorithm',   'LGBMClassifier'),
        ('Tuning',      'FLAML 180s'),
        ('n_estimators', params.get('n_estimators', 799)),
        ('num_leaves',  params.get('num_leaves', 17)),
        ('Calibration', 'Platt (X_val)'),
        ('Features',    '18 (15 cat + 3 num)'),
        ('Train rows',  f"{meta.get('train_rows',5282):,}"),
        ('θ_optimal',   meta.get('theta_optimal', 0.41)),
        ('Vc / Cr',     f"${meta.get('Vc',69)} / ${meta.get('Cr',20)}"),
    ]
    rows_html = ''.join(
        f"<tr><td style='color:#6B7A99;padding:5px 0;width:48%'>{k}</td>"
        f"<td style='font-family:IBM Plex Mono,monospace;'>{v}</td></tr>"
        for k, v in rows_mc
    )
    st.markdown(f"""
    <div class='section-card'>
        <div class='section-title'>Model Card</div>
        <table style='width:100%;font-size:12px;border-collapse:collapse;'>
            {rows_html}
        </table>
    </div>
    """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────
st.markdown("""
<hr style='border-color:#1E2D45;margin-top:20px;'>
<div style='text-align:center;font-size:11px;color:#6B7A99;
            font-family:IBM Plex Mono,monospace;padding-bottom:14px;'>
    Interconnect Telecom · Churn Intelligence System · LightGBM + Platt + SHAP · AUC 0.9078
</div>
""", unsafe_allow_html=True)
