"""
╔══════════════════════════════════════════════════════════════════╗
║   MERCHANT RISK ANALYTICS DASHBOARD                              ║
║   Onboarding Collusion Detection – Affirm-Style                  ║
║   Dataset: IEEE-CIS Fraud + PaySim patterns (Kaggle-inspired)    ║
╚══════════════════════════════════════════════════════════════════╝
Como rodar:
    pip install streamlit pandas numpy scikit-learn matplotlib seaborn
    streamlit run dashboard_affirm.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Merchant Risk Analytics – Affirm Style",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CUSTOM CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

  html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
    background-color: #0b0f1a;
    color: #e2e8f0;
  }
  .main { background-color: #0b0f1a; }
  .block-container { padding-top: 1.5rem; }

  /* KPI Cards */
  .kpi-card {
    background: linear-gradient(135deg, #141c2e 0%, #1a2540 100%);
    border: 1px solid #2d3a52;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 8px;
  }
  .kpi-label { font-size: 11px; font-weight: 600; letter-spacing: 0.12em;
               text-transform: uppercase; color: #64748b; margin-bottom: 4px; }
  .kpi-value { font-family: 'IBM Plex Mono', monospace; font-size: 32px;
               font-weight: 600; color: #f1f5f9; line-height: 1; }
  .kpi-delta-pos { font-size: 12px; color: #34d399; margin-top: 4px; }
  .kpi-delta-neg { font-size: 12px; color: #f87171; margin-top: 4px; }

  /* Section headers */
  .section-header {
    font-size: 13px; font-weight: 700; letter-spacing: 0.1em;
    text-transform: uppercase; color: #7c8db5;
    border-bottom: 1px solid #1e2d45;
    padding-bottom: 6px; margin-bottom: 16px; margin-top: 28px;
  }

  /* Alert box */
  .alert-high {
    background: rgba(239,68,68,0.08); border-left: 3px solid #ef4444;
    border-radius: 6px; padding: 12px 16px; margin: 6px 0;
    font-size: 13px;
  }
  .alert-med {
    background: rgba(245,158,11,0.08); border-left: 3px solid #f59e0b;
    border-radius: 6px; padding: 12px 16px; margin: 6px 0;
    font-size: 13px;
  }
  .stDataFrame { border-radius: 8px; }
  h1 { font-family: 'IBM Plex Sans'; font-weight: 700; color: #f1f5f9; }
  h2, h3 { font-family: 'IBM Plex Sans'; color: #cbd5e1; }
</style>
""", unsafe_allow_html=True)

# ── PALETTE ───────────────────────────────────────────────────────────────────
CLR = {
    "bg":      "#0b0f1a",
    "panel":   "#141c2e",
    "border":  "#2d3a52",
    "accent":  "#3b82f6",
    "danger":  "#ef4444",
    "warn":    "#f59e0b",
    "ok":      "#34d399",
    "text":    "#e2e8f0",
    "muted":   "#64748b",
}
PALETTE_RISK = [CLR["ok"], CLR["warn"], CLR["danger"]]


# ── DATA ──────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    """Gera ou carrega o dataset de merchant risk."""
    import os
    path = "merchant_risk_dataset.csv"
    if os.path.exists(path):
        return pd.read_csv(path)

    # ── GENERATE SYNTHETIC DATA (Kaggle IEEE-CIS + PaySim inspired) ──────────
    np.random.seed(42)
    N = 5000
    MCC = {
        "5411":"Grocery","5812":"Restaurants","5999":"Misc Retail",
        "7372":"Software/SaaS","5045":"Electronics Wholesale",
        "5065":"Electronics Retail","7011":"Hotels","5912":"Drug Stores",
        "7995":"Gambling","6012":"Financial Services",
        "5961":"Mail Order","5094":"Precious Metals",
    }
    HIGH_RISK_MCC = {"7995","6012","5961","5094","5065"}
    COUNTRIES = ["US","CA","GB","MX","BR","NG","CN","IN","RO","VN"]
    HR_COUNTRIES = {"NG","RO","VN","MX"}
    FRAUD_REASONS = [
        "Collusion – Merchant-Consumer Ring","Bust-Out Scheme",
        "Identity Fraud at Onboarding","Triangulation Fraud",
        "Chargeback Abuse","Money Laundering","Synthetic Identity",
        "Card Testing","Friendly Fraud","Account Takeover",
    ]

    mcc_keys = list(MCC.keys())
    mcc_w = np.array([0.12,0.15,0.10,0.08,0.06,0.07,0.06,0.08,0.05,0.06,0.07,0.05])
    mcc_w /= mcc_w.sum()
    c_w = np.array([0.30,0.15,0.10,0.08,0.07,0.07,0.05,0.05,0.05,0.03]); c_w /= c_w.sum()

    mcc_codes      = np.random.choice(mcc_keys, N, p=mcc_w)
    countries_arr  = np.random.choice(COUNTRIES, N, p=c_w)
    days_onboard   = np.random.exponential(180, N).clip(1,730).astype(int)
    monthly_vol    = np.random.lognormal(9,1.5,N).round(2)
    avg_ticket     = np.random.lognormal(4.5,1.0,N).round(2)
    txn_velocity   = np.random.lognormal(3,1.2,N).round(1)
    cb_rate        = (np.random.beta(1.2,20,N)*10).round(3)
    refund_rate    = (np.random.beta(1.5,15,N)*15).round(3)
    hr_bin_pct     = (np.random.beta(1,5,N)*100).round(2)
    intl_pct       = (np.random.beta(1,8,N)*100).round(2)
    id_risk        = (np.random.beta(1.5,4,N)*100).round(1)
    hrs_first      = np.random.exponential(48,N).clip(0.1,500).round(2)
    s_bank  = np.random.choice([0,1],N,p=[0.92,0.08])
    s_ip    = np.random.choice([0,1],N,p=[0.88,0.12])
    s_phone = np.random.choice([0,1],N,p=[0.94,0.06])
    s_owner = np.random.choice([0,1],N,p=[0.90,0.10])
    col_sc  = s_bank+s_ip+s_phone+s_owner
    net_sz  = np.where(col_sc>=2, np.random.randint(2,15,N), 1)
    vel_spk = np.random.choice([0,1],N,p=[0.85,0.15])

    probs = np.zeros(N)
    for i in range(N):
        s=0.0
        if mcc_codes[i] in HIGH_RISK_MCC: s+=0.22
        if countries_arr[i] in HR_COUNTRIES: s+=0.18
        s+=cb_rate[i]*0.07; s+=col_sc[i]*0.14
        if net_sz[i]>=5: s+=0.18
        if hrs_first[i]<2: s+=0.15
        s+=intl_pct[i]*0.003; s+=id_risk[i]*0.004; s+=hr_bin_pct[i]*0.002
        if days_onboard[i]<30: s+=0.10
        if vel_spk[i]: s+=0.12
        probs[i]=min(s,1.0)
    probs=(probs+np.random.normal(0,0.04,N)).clip(0,1)
    is_fraud=(np.random.random(N)<probs).astype(int)

    def rsn(i):
        if not is_fraud[i]: return "None"
        if col_sc[i]>=2: return np.random.choice(["Collusion – Merchant-Consumer Ring","Money Laundering","Bust-Out Scheme"],p=[0.5,0.3,0.2])
        if cb_rate[i]>4: return np.random.choice(["Chargeback Abuse","Friendly Fraud","Triangulation Fraud"],p=[0.5,0.3,0.2])
        if id_risk[i]>60: return np.random.choice(["Identity Fraud at Onboarding","Synthetic Identity","Account Takeover"],p=[0.5,0.3,0.2])
        if hrs_first[i]<3: return np.random.choice(["Card Testing","Account Takeover","Bust-Out Scheme"],p=[0.5,0.3,0.2])
        return np.random.choice(FRAUD_REASONS)

    from datetime import datetime, timedelta
    start = datetime(2023,1,1)
    dates = [(start+timedelta(days=int(np.random.randint(0,700)))).strftime("%Y-%m-%d") for _ in range(N)]
    rscr  = (probs*100).round(1)
    rtier = pd.cut(rscr,bins=[0,30,60,100],labels=["Low","Medium","High"])

    df = pd.DataFrame({
        "merchant_id":[f"MER{str(i).zfill(5)}" for i in range(N)],
        "onboarding_date":dates,"days_since_onboarding":days_onboard,
        "mcc_code":mcc_codes,"mcc_description":[MCC[m] for m in mcc_codes],
        "country":countries_arr,"monthly_volume_usd":monthly_vol,
        "avg_ticket_usd":avg_ticket,"txn_velocity_per_day":txn_velocity,
        "chargeback_rate_pct":cb_rate,"refund_rate_pct":refund_rate,
        "high_risk_bin_pct":hr_bin_pct,"intl_card_pct":intl_pct,
        "identity_risk_score":id_risk,"hours_to_first_txn":hrs_first,
        "shared_bank_account":s_bank,"shared_ip_address":s_ip,
        "shared_phone":s_phone,"shared_owner_name":s_owner,
        "collusion_score":col_sc,"network_size":net_sz,
        "velocity_spike":vel_spk,"fraud_risk_score":rscr,
        "risk_tier":rtier,"is_fraud":is_fraud,
        "fraud_reason":[rsn(i) for i in range(N)],
    })
    return df

@st.cache_resource
def train_model(df):
    FEATURES = [
        "chargeback_rate_pct","refund_rate_pct","high_risk_bin_pct","intl_card_pct",
        "identity_risk_score","hours_to_first_txn","collusion_score","network_size",
        "velocity_spike","days_since_onboarding","monthly_volume_usd","avg_ticket_usd",
        "txn_velocity_per_day","shared_bank_account","shared_ip_address",
        "shared_phone","shared_owner_name",
    ]
    le = LabelEncoder()
    X = df[FEATURES].copy()
    y = df["is_fraud"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    clf = RandomForestClassifier(n_estimators=120, max_depth=8, random_state=42, n_jobs=-1)
    clf.fit(X_tr, y_tr)
    auc = roc_auc_score(y_te, clf.predict_proba(X_te)[:,1])
    imp = pd.Series(clf.feature_importances_, index=FEATURES).sort_values(ascending=False)
    return clf, auc, imp, FEATURES

# ── CHART HELPERS ─────────────────────────────────────────────────────────────
def dark_fig(w=8, h=4):
    fig, ax = plt.subplots(figsize=(w,h))
    fig.patch.set_facecolor(CLR["panel"])
    ax.set_facecolor(CLR["panel"])
    ax.tick_params(colors=CLR["muted"], labelsize=9)
    ax.xaxis.label.set_color(CLR["muted"])
    ax.yaxis.label.set_color(CLR["muted"])
    for spine in ax.spines.values():
        spine.set_edgecolor(CLR["border"])
    return fig, ax

def kpi(label, value, delta=None, delta_positive=True):
    delta_html = ""
    if delta:
        cls = "kpi-delta-pos" if delta_positive else "kpi-delta-neg"
        arrow = "↑" if delta_positive else "↓"
        delta_html = f'<div class="{cls}">{arrow} {delta}</div>'
    st.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{value}</div>
      {delta_html}
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APP
# ══════════════════════════════════════════════════════════════════════════════
df_raw = load_data()

# ── SIDEBAR FILTERS ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ Merchant Risk")
    st.markdown("**Analyst I – Onboarding Collusion**")
    st.markdown("---")

    st.markdown("**Risk Tier**")
    tier_filter = st.multiselect("", ["Low","Medium","High"],
                                  default=["Low","Medium","High"], label_visibility="collapsed")

    st.markdown("**Country**")
    country_filter = st.multiselect("", sorted(df_raw["country"].unique()),
                                     default=sorted(df_raw["country"].unique()), label_visibility="collapsed")

    st.markdown("**MCC Category**")
    mcc_filter = st.multiselect("", sorted(df_raw["mcc_description"].unique()),
                                  default=sorted(df_raw["mcc_description"].unique()), label_visibility="collapsed")

    min_score, max_score = st.slider("Fraud Risk Score", 0, 100, (0, 100))
    st.markdown("---")
    st.caption("Dataset: IEEE-CIS Fraud + PaySim patterns (Kaggle-inspired synthetic)")

# ── APPLY FILTERS ────────────────────────────────────────────────────────────
df = df_raw.copy()
df = df[df["risk_tier"].isin(tier_filter)]
df = df[df["country"].isin(country_filter)]
df = df[df["mcc_description"].isin(mcc_filter)]
df = df[(df["fraud_risk_score"] >= min_score) & (df["fraud_risk_score"] <= max_score)]

# ── TRAIN MODEL ──────────────────────────────────────────────────────────────
clf, auc, feat_imp, FEATURES = train_model(df_raw)

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("# 🛡️ Merchant Risk Analytics Dashboard")
st.markdown(
    f"**Onboarding Collusion Detection** · {len(df):,} merchants in view · "
    f"Data: IEEE-CIS Fraud + PaySim (Kaggle-style) · Model AUC: **{auc:.3f}**"
)
st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
# KPI ROW
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">📊 KEY METRICS</div>', unsafe_allow_html=True)
k1,k2,k3,k4,k5 = st.columns(5)
total     = len(df)
fraud_ct  = df["is_fraud"].sum()
fraud_pct = fraud_ct/total*100 if total else 0
high_risk = (df["risk_tier"]=="High").sum()
collusion_merchants = (df["collusion_score"] >= 2).sum()
avg_cb    = df["chargeback_rate_pct"].mean()

with k1: kpi("Total Merchants", f"{total:,}")
with k2: kpi("Fraud Detected", f"{fraud_ct:,}", f"{fraud_pct:.1f}% of portfolio", delta_positive=False)
with k3: kpi("High-Risk Tier", f"{high_risk:,}", f"{high_risk/total*100:.1f}% of total", delta_positive=False)
with k4: kpi("Collusion Signals", f"{collusion_merchants:,}", "2+ shared attributes", delta_positive=False)
with k5: kpi("Avg Chargeback Rate", f"{avg_cb:.2f}%")

# ══════════════════════════════════════════════════════════════════════════════
# ROW 1: Fraud Reasons + Risk Tier Distribution
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">🚨 FRAUD PAINPOINTS & DISTRIBUTION</div>', unsafe_allow_html=True)
col1, col2 = st.columns([3, 2])

with col1:
    fraud_df = df[df["is_fraud"]==1]
    if len(fraud_df) > 0:
        reason_ct = fraud_df["fraud_reason"].value_counts().head(10)
        fig, ax = dark_fig(8, 4)
        colors = [CLR["danger"] if "Collusion" in r or "Money" in r else
                  CLR["warn"]   if "Chargeback" in r or "Bust" in r else
                  CLR["accent"] for r in reason_ct.index]
        bars = ax.barh(reason_ct.index[::-1], reason_ct.values[::-1], color=colors[::-1], edgecolor="none", height=0.65)
        ax.set_xlabel("Number of Merchants", color=CLR["muted"])
        ax.set_title("Top Fraud Reasons", color=CLR["text"], fontsize=12, fontweight="bold", pad=10)
        for bar, val in zip(bars, reason_ct.values[::-1]):
            ax.text(val+2, bar.get_y()+bar.get_height()/2, str(val),
                    va="center", ha="left", color=CLR["muted"], fontsize=8)
        patches = [mpatches.Patch(color=CLR["danger"],label="Collusion/Laundering"),
                   mpatches.Patch(color=CLR["warn"],label="Financial Abuse"),
                   mpatches.Patch(color=CLR["accent"],label="Identity/Tech Fraud")]
        ax.legend(handles=patches, loc="lower right", facecolor=CLR["panel"],
                  edgecolor=CLR["border"], labelcolor=CLR["muted"], fontsize=8)
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()
    else:
        st.info("No fraud records in current filter.")

with col2:
    tier_ct = df["risk_tier"].value_counts().reindex(["Low","Medium","High"]).fillna(0)
    fig, ax = dark_fig(5, 4)
    wedge_colors = [CLR["ok"], CLR["warn"], CLR["danger"]]
    wedges, texts, autotexts = ax.pie(
        tier_ct.values, labels=tier_ct.index,
        colors=wedge_colors, autopct="%1.1f%%",
        startangle=140, pctdistance=0.78,
        wedgeprops=dict(edgecolor=CLR["bg"], linewidth=2)
    )
    for t in texts: t.set_color(CLR["text"]); t.set_fontsize(10)
    for a in autotexts: a.set_color(CLR["bg"]); a.set_fontsize(9); a.set_fontweight("bold")
    ax.set_title("Risk Tier Split", color=CLR["text"], fontsize=12, fontweight="bold", pad=10)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# ROW 2: Collusion Heatmap + Feature Importance
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">🕸️ COLLUSION SIGNALS & MODEL INSIGHTS</div>', unsafe_allow_html=True)
col3, col4 = st.columns(2)

with col3:
    # Collusion signals breakdown
    cols_signals = ["shared_bank_account","shared_ip_address","shared_phone","shared_owner_name","velocity_spike"]
    labels       = ["Shared Bank","Shared IP","Shared Phone","Shared Owner","Velocity Spike"]
    fraud_rates  = [(df[df[c]==1]["is_fraud"].mean()*100) if df[c].sum()>0 else 0 for c in cols_signals]
    counts       = [df[c].sum() for c in cols_signals]

    fig, ax = dark_fig(6, 4)
    x = np.arange(len(labels))
    bar_w = 0.4
    bars1 = ax.bar(x - bar_w/2, counts, bar_w, color=CLR["accent"], alpha=0.85, label="Count of Merchants")
    ax2 = ax.twinx()
    ax2.set_facecolor(CLR["panel"])
    bars2 = ax2.bar(x + bar_w/2, fraud_rates, bar_w, color=CLR["danger"], alpha=0.85, label="Fraud Rate %")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right", color=CLR["muted"], fontsize=8)
    ax.set_ylabel("# Merchants with Signal", color=CLR["accent"], fontsize=9)
    ax2.set_ylabel("Fraud Rate %", color=CLR["danger"], fontsize=9)
    ax2.tick_params(colors=CLR["muted"])
    ax2.spines["right"].set_edgecolor(CLR["border"])
    for spine in ax2.spines.values(): spine.set_edgecolor(CLR["border"])
    ax.set_title("Collusion Signals: Volume vs Fraud Rate", color=CLR["text"], fontsize=11, fontweight="bold", pad=10)
    lines = [mpatches.Patch(color=CLR["accent"],label="Count"),
             mpatches.Patch(color=CLR["danger"],label="Fraud %")]
    ax.legend(handles=lines, loc="upper right", facecolor=CLR["panel"],
              edgecolor=CLR["border"], labelcolor=CLR["muted"], fontsize=8)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

with col4:
    # Feature Importance from RF model
    top_feats = feat_imp.head(12)
    fig, ax = dark_fig(6, 4)
    clrs = [CLR["danger"] if "collusion" in f or "network" in f or "shared" in f else
            CLR["warn"]   if "chargeback" in f or "identity" in f else
            CLR["accent"] for f in top_feats.index]
    ax.barh(top_feats.index[::-1], top_feats.values[::-1], color=clrs[::-1], edgecolor="none", height=0.65)
    ax.set_xlabel("Feature Importance", color=CLR["muted"], fontsize=9)
    ax.set_title("🤖 Model: Top Risk Drivers", color=CLR["text"], fontsize=11, fontweight="bold", pad=10)
    ax.tick_params(labelsize=8)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# ROW 3: Fraud Rate by Country + MCC
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">🌍 GEO & CATEGORY RISK BREAKDOWN</div>', unsafe_allow_html=True)
col5, col6 = st.columns(2)

with col5:
    country_grp = df.groupby("country").agg(
        total=("is_fraud","count"), fraud=("is_fraud","sum")
    ).reset_index()
    country_grp["fraud_rate"] = country_grp["fraud"]/country_grp["total"]*100
    country_grp = country_grp.sort_values("fraud_rate", ascending=True)

    fig, ax = dark_fig(6,4)
    bar_colors = [CLR["danger"] if r > 50 else CLR["warn"] if r > 30 else CLR["ok"]
                  for r in country_grp["fraud_rate"]]
    ax.barh(country_grp["country"], country_grp["fraud_rate"], color=bar_colors, edgecolor="none", height=0.65)
    ax.axvline(x=country_grp["fraud_rate"].mean(), color=CLR["muted"], linestyle="--", alpha=0.5, linewidth=1)
    ax.set_xlabel("Fraud Rate %", color=CLR["muted"])
    ax.set_title("Fraud Rate by Country", color=CLR["text"], fontsize=11, fontweight="bold", pad=10)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

with col6:
    mcc_grp = df.groupby("mcc_description").agg(
        total=("is_fraud","count"), fraud=("is_fraud","sum")
    ).reset_index()
    mcc_grp["fraud_rate"] = mcc_grp["fraud"]/mcc_grp["total"]*100
    mcc_grp = mcc_grp.sort_values("fraud_rate", ascending=True)

    fig, ax = dark_fig(6,4)
    bar_colors = [CLR["danger"] if r > 55 else CLR["warn"] if r > 35 else CLR["ok"]
                  for r in mcc_grp["fraud_rate"]]
    ax.barh(mcc_grp["mcc_description"], mcc_grp["fraud_rate"], color=bar_colors, edgecolor="none", height=0.65)
    ax.axvline(x=mcc_grp["fraud_rate"].mean(), color=CLR["muted"], linestyle="--", alpha=0.5, linewidth=1)
    ax.set_xlabel("Fraud Rate %", color=CLR["muted"])
    ax.set_title("Fraud Rate by MCC Category", color=CLR["text"], fontsize=11, fontweight="bold", pad=10)
    ax.tick_params(labelsize=8)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# ROW 4: Onboarding Risk: Hours to First Txn + Score Distribution
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">⏱️ ONBOARDING RISK SIGNALS</div>', unsafe_allow_html=True)
col7, col8 = st.columns(2)

with col7:
    fig, ax = dark_fig(6,4)
    fraud_hrs   = df[df["is_fraud"]==1]["hours_to_first_txn"].clip(upper=200)
    legit_hrs   = df[df["is_fraud"]==0]["hours_to_first_txn"].clip(upper=200)
    ax.hist(legit_hrs,  bins=40, color=CLR["ok"],     alpha=0.6, label="Legitimate", density=True)
    ax.hist(fraud_hrs,  bins=40, color=CLR["danger"],  alpha=0.6, label="Fraud",      density=True)
    ax.axvline(x=12, color=CLR["warn"], linestyle="--", linewidth=1.2, label="12h threshold")
    ax.set_xlabel("Hours to First Transaction After Onboarding", color=CLR["muted"])
    ax.set_ylabel("Density", color=CLR["muted"])
    ax.set_title("Onboarding Speed: Fraud vs Legit", color=CLR["text"], fontsize=11, fontweight="bold", pad=10)
    ax.legend(facecolor=CLR["panel"], edgecolor=CLR["border"], labelcolor=CLR["muted"], fontsize=9)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

with col8:
    fig, ax = dark_fig(6,4)
    scores_legit = df[df["is_fraud"]==0]["fraud_risk_score"]
    scores_fraud = df[df["is_fraud"]==1]["fraud_risk_score"]
    ax.hist(scores_legit, bins=40, color=CLR["ok"],    alpha=0.6, label="Legitimate", density=True)
    ax.hist(scores_fraud, bins=40, color=CLR["danger"], alpha=0.6, label="Fraud",      density=True)
    ax.set_xlabel("Fraud Risk Score (0–100)", color=CLR["muted"])
    ax.set_ylabel("Density", color=CLR["muted"])
    ax.set_title("Risk Score Distribution", color=CLR["text"], fontsize=11, fontweight="bold", pad=10)
    ax.legend(facecolor=CLR["panel"], edgecolor=CLR["border"], labelcolor=CLR["muted"], fontsize=9)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# HIGH-RISK MERCHANT TABLE
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">🔴 HIGH-RISK MERCHANT WATCHLIST</div>', unsafe_allow_html=True)

top_risky = df[df["risk_tier"]=="High"].sort_values("fraud_risk_score", ascending=False).head(25)
if len(top_risky) == 0:
    top_risky = df.sort_values("fraud_risk_score", ascending=False).head(25)

display_cols = ["merchant_id","country","mcc_description","fraud_risk_score","collusion_score",
                "chargeback_rate_pct","hours_to_first_txn","fraud_reason","is_fraud"]
top_disp = top_risky[display_cols].rename(columns={
    "merchant_id":"Merchant ID","country":"Country","mcc_description":"Category",
    "fraud_risk_score":"Risk Score","collusion_score":"Collusion Score",
    "chargeback_rate_pct":"CB Rate %","hours_to_first_txn":"Hrs to 1st Txn",
    "fraud_reason":"Fraud Reason","is_fraud":"Fraud Flag",
})

def color_risk(val):
    if val >= 70: return "color: #ef4444; font-weight: bold"
    if val >= 40: return "color: #f59e0b"
    return "color: #34d399"

def color_fraud(val):
    return "color: #ef4444; font-weight: bold" if val == 1 else "color: #34d399"

styled = top_disp.style\
    .applymap(color_risk, subset=["Risk Score"])\
    .applymap(color_fraud, subset=["Fraud Flag"])\
    .set_properties(**{"background-color": CLR["panel"], "border": f"1px solid {CLR['border']}"})\
    .format({"Risk Score":"{:.1f}","CB Rate %":"{:.3f}","Hrs to 1st Txn":"{:.1f}"})

st.dataframe(styled, use_container_width=True, height=380)

# ══════════════════════════════════════════════════════════════════════════════
# LIVE MERCHANT SCORER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">⚡ REAL-TIME MERCHANT RISK SCORER</div>', unsafe_allow_html=True)
st.markdown("Simule o onboarding de um novo merchant e veja o score de risco ao vivo:")

sc1, sc2, sc3, sc4 = st.columns(4)
with sc1:
    inp_cb     = st.number_input("Chargeback Rate %",    0.0, 15.0, 1.5, 0.1)
    inp_col    = st.slider("Collusion Score (0-4)",       0, 4, 0)
    inp_id     = st.slider("Identity Risk Score (0-100)", 0, 100, 20)
with sc2:
    inp_hrs    = st.number_input("Hours to First Txn",   0.1, 500.0, 48.0, 1.0)
    inp_net    = st.slider("Network Size",                1, 15, 1)
    inp_intl   = st.slider("Intl Card %",                 0, 100, 10)
with sc3:
    inp_hr_bin = st.slider("High-Risk BIN %",             0, 100, 5)
    inp_days   = st.number_input("Days Since Onboarding", 1, 730, 60, 1)
    inp_vel    = st.selectbox("Velocity Spike?",          ["No","Yes"])
with sc4:
    inp_refund = st.number_input("Refund Rate %",         0.0, 20.0, 2.0, 0.1)
    inp_vol    = st.number_input("Monthly Volume (USD)",  100.0, 500000.0, 10000.0, 500.0)
    inp_ticket = st.number_input("Avg Ticket (USD)",      1.0, 5000.0, 50.0, 5.0)

if st.button("🔍  Score This Merchant", type="primary"):
    sample = pd.DataFrame([{
        "chargeback_rate_pct":  inp_cb,
        "refund_rate_pct":      inp_refund,
        "high_risk_bin_pct":    inp_hr_bin,
        "intl_card_pct":        inp_intl,
        "identity_risk_score":  inp_id,
        "hours_to_first_txn":   inp_hrs,
        "collusion_score":      inp_col,
        "network_size":         inp_net,
        "velocity_spike":       1 if inp_vel=="Yes" else 0,
        "days_since_onboarding":inp_days,
        "monthly_volume_usd":   inp_vol,
        "avg_ticket_usd":       inp_ticket,
        "txn_velocity_per_day": 50,
        "shared_bank_account":  min(inp_col,1),
        "shared_ip_address":    1 if inp_col>=2 else 0,
        "shared_phone":         1 if inp_col>=3 else 0,
        "shared_owner_name":    1 if inp_col>=4 else 0,
    }])
    proba = clf.predict_proba(sample[FEATURES])[0][1]
    score = round(proba*100, 1)
    tier  = "🔴 HIGH" if score>=60 else "🟡 MEDIUM" if score>=30 else "🟢 LOW"

    r1, r2, r3 = st.columns(3)
    with r1:
        st.markdown(f"""<div class="kpi-card">
          <div class="kpi-label">FRAUD PROBABILITY</div>
          <div class="kpi-value" style="color:{'#ef4444' if score>=60 else '#f59e0b' if score>=30 else '#34d399'}">{score}%</div>
        </div>""", unsafe_allow_html=True)
    with r2:
        st.markdown(f"""<div class="kpi-card">
          <div class="kpi-label">RISK TIER</div>
          <div class="kpi-value" style="font-size:20px;padding-top:8px">{tier}</div>
        </div>""", unsafe_allow_html=True)
    with r3:
        action = "🚫 DECLINE / MANUAL REVIEW" if score>=60 else "⚠️ ENHANCED MONITORING" if score>=30 else "✅ APPROVE WITH STANDARD LIMITS"
        st.markdown(f"""<div class="kpi-card">
          <div class="kpi-label">RECOMMENDED ACTION</div>
          <div class="kpi-value" style="font-size:14px;padding-top:8px;line-height:1.4">{action}</div>
        </div>""", unsafe_allow_html=True)

    # Top risk drivers for this merchant
    sample_vals = sample[FEATURES].values[0]
    drivers = pd.Series(np.abs(sample_vals * feat_imp.values), index=feat_imp.index).sort_values(ascending=False).head(5)
    st.markdown("**Top risk drivers for this merchant:**")
    for feat, val in drivers.items():
        bar_len = int(val / drivers.max() * 200)
        st.markdown(f"`{feat}` &nbsp; <span style='display:inline-block;background:#ef4444;width:{bar_len}px;height:8px;border-radius:4px;vertical-align:middle'></span>", unsafe_allow_html=True)

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<small style='color:#475569'>Built for Affirm Merchant Risk Analytics · "
    "Model: Random Forest (AUC {:.3f}) · ".format(auc) +
    "Dataset: Kaggle IEEE-CIS Fraud + PaySim patterns · 5,000 synthetic merchants</small>",
    unsafe_allow_html=True
)
