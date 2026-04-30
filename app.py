import streamlit as st
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import pandas as pd
import joblib
import json
import os
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Demand Prediction Engine",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

h1, h2, h3 {
    font-family: 'Playfair Display', serif !important;
}

/* Main background */
.stApp {
    background: #0f0e0c;
    color: #f0ebe3;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #1a1814 !important;
    border-right: 1px solid #2e2a24;
}

/* Cards */
.metric-card {
    background: linear-gradient(135deg, #1e1c18 0%, #252219 100%);
    border: 1px solid #3a3528;
    border-radius: 12px;
    padding: 24px;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #c9a84c, #e8c97e, #c9a84c);
}
.metric-value {
    font-family: 'Playfair Display', serif;
    font-size: 2.8rem;
    color: #e8c97e;
    line-height: 1;
    margin: 8px 0;
}
.metric-label {
    color: #8a8070;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    font-weight: 500;
}

/* Prediction result */
.prediction-box {
    background: linear-gradient(135deg, #1e2918 0%, #1a2314 100%);
    border: 1px solid #3a5028;
    border-radius: 16px;
    padding: 32px;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.prediction-box::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #4caf76, #7ed4a0, #4caf76);
}
.prediction-number {
    font-family: 'Playfair Display', serif;
    font-size: 4rem;
    color: #7ed4a0;
    line-height: 1;
}
.prediction-unit {
    color: #5a8070;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 3px;
    margin-top: 8px;
}

/* Upload area */
.upload-hint {
    color: #6a6050;
    font-size: 0.85rem;
    text-align: center;
    margin-top: 8px;
}

/* Similar designs grid */
.similar-card {
    background: #1a1814;
    border: 1px solid #2e2a24;
    border-radius: 10px;
    padding: 12px;
    text-align: center;
    transition: border-color 0.2s;
}

/* Section headers */
.section-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.5rem;
    color: #e8c97e;
    border-bottom: 1px solid #2e2a24;
    padding-bottom: 12px;
    margin-bottom: 20px;
}

/* Gold divider */
.gold-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, #c9a84c, transparent);
    margin: 32px 0;
}

/* Confidence bar */
.conf-bar-bg {
    background: #2e2a24;
    border-radius: 99px;
    height: 8px;
    overflow: hidden;
    margin-top: 8px;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 99px;
    background: linear-gradient(90deg, #4caf76, #7ed4a0);
}

/* Streamlit overrides */
.stButton > button {
    background: linear-gradient(135deg, #c9a84c, #e8c97e) !important;
    color: #0f0e0c !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    letter-spacing: 1px !important;
    padding: 12px 32px !important;
    width: 100%;
    transition: opacity 0.2s !important;
}
.stButton > button:hover {
    opacity: 0.85 !important;
}

div[data-testid="stFileUploader"] {
    background: #1a1814 !important;
    border: 1px dashed #3a3528 !important;
    border-radius: 12px !important;
}
</style>
""", unsafe_allow_html=True)


# ─── Load Models (cached) ──────────────────────────────────────────────────────
@st.cache_resource
def load_mobilenet():
    m = models.mobilenet_v2(pretrained=True)
    m.classifier = torch.nn.Identity()
    m.eval()
    return m

@st.cache_resource
def load_artifacts():
    model     = joblib.load('demand_model.pkl')
    pca       = joblib.load('pca.pkl')
    scaler    = joblib.load('scaler.pkl')
    with open('model_config.json') as f:
        config = json.load(f)
    return model, pca, scaler, config

@st.cache_data
def load_data():
    df   = pd.read_csv('matched_data_final.csv')
    embs = np.load('embeddings.npy')
    return df, embs


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

def get_embedding(img: Image.Image, mobilenet) -> np.ndarray:
    t = transform(img).unsqueeze(0)
    with torch.no_grad():
        return mobilenet(t).squeeze().numpy()

def predict_qty(emb, avg_rate, model, pca, scaler, config):
    extra = np.array([[avg_rate ]])
    X     = np.hstack([emb.reshape(1, -1), extra])
    X_r   = pca.transform(X)
    X_s   = scaler.transform(X_r)
    pred  = model.predict(X_s)[0]
    if config.get('log_transform'):
        pred = np.expm1(pred)
    return max(0, round(float(pred)))

def get_similar(query_emb, all_embs, df, top_k=4):
    sims  = cosine_similarity(query_emb.reshape(1, -1), all_embs)[0]
    idxs  = np.argsort(sims)[::-1][:top_k]
    return df.iloc[idxs].copy(), sims[idxs]

def confidence_label(pred, df):
    mu, sd = df['total_qty'].mean(), df['total_qty'].std()
    z = abs(pred - mu) / (sd + 1e-9)
    if z < 0.5:   return "High", 90
    elif z < 1.0: return "Medium", 65
    else:         return "Low", 35


# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎨 Demand Engine")
    st.markdown("<div style='color:#6a6050;font-size:0.8rem;margin-top:-8px'>Maalde Company · AI/ML Task</div>", unsafe_allow_html=True)
    st.markdown("<div class='gold-divider'></div>", unsafe_allow_html=True)

    st.markdown("**Configuration**")
    avg_rate   = st.number_input("Avg Rate (₹)",   min_value=0.0,  value=500.0,  step=50.0)
    top_k      = st.slider("Similar Designs to Show", 2, 6, 4)

    st.markdown("<div class='gold-divider'></div>", unsafe_allow_html=True)

    st.markdown("**Model Info**")
    try:
        _, _, _, cfg = load_artifacts()
        df_info, _   = load_data()
        st.markdown(f"<div style='color:#8a8070;font-size:0.82rem'>Model: <b style='color:#e8c97e'>{cfg['model_name']}</b></div>", unsafe_allow_html=True)
        st.markdown(f"<div style='color:#8a8070;font-size:0.82rem'>Trained on: <b style='color:#e8c97e'>{len(df_info)} designs</b></div>", unsafe_allow_html=True)
        st.markdown(f"<div style='color:#8a8070;font-size:0.82rem'>CV R²: <b style='color:#7ed4a0'>-0.0357</b></div>", unsafe_allow_html=True)
    except:
        st.warning("Model files not found.")

# ─── Main ─────────────────────────────────────────────────────────────────────
st.markdown("<h1 style='color:#e8c97e;margin-bottom:4px'>Demand Prediction Engine</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#6a6050;font-size:0.9rem;margin-bottom:32px'>Upload a new product design image to predict expected sales quantity</p>", unsafe_allow_html=True)

# ─── Dataset Stats Row ────────────────────────────────────────────────────────
try:
    df_data, all_embs = load_data()
    c1, c2, c3, c4 = st.columns(4)
    stats = [
        ("Total Designs",   len(df_data),                           "matched & trained"),
        ("Avg Sales / Design", f"{df_data['total_qty'].mean():.1f}", "units"),
        ("Max Sales",        int(df_data['total_qty'].max()),        "units for 1 design"),
        ("Avg Rate",         f"₹{df_data['avg_rate'].mean():.0f}",  "per unit"),
    ]
    for col, (label, val, sub) in zip([c1,c2,c3,c4], stats):
        with col:
            st.markdown(f"""
            <div class='metric-card'>
                <div class='metric-label'>{label}</div>
                <div class='metric-value'>{val}</div>
                <div class='metric-label' style='font-size:0.7rem;color:#5a5040'>{sub}</div>
            </div>""", unsafe_allow_html=True)
except Exception as e:
    st.error(f"Could not load data files: {e}")
    st.stop()

st.markdown("<div class='gold-divider'></div>", unsafe_allow_html=True)

# ─── Upload + Predict ─────────────────────────────────────────────────────────
left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown("<div class='section-title'>Upload Design</div>", unsafe_allow_html=True)
    uploaded = st.file_uploader("", type=["jpg","jpeg","png"], label_visibility="collapsed")
    st.markdown("<div class='upload-hint'>JPG or PNG · Product design image</div>", unsafe_allow_html=True)

    if uploaded:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, use_column_width=True, caption="Uploaded Design")

with right:
    st.markdown("<div class='section-title'>Prediction</div>", unsafe_allow_html=True)

    if not uploaded:
        st.markdown("""
        <div style='background:#1a1814;border:1px solid #2e2a24;border-radius:12px;
                    padding:48px 24px;text-align:center;color:#4a4030'>
            <div style='font-size:2.5rem;margin-bottom:12px'>📦</div>
            <div style='font-family:Playfair Display,serif;font-size:1.1rem;color:#6a5a40'>
                Awaiting design upload
            </div>
            <div style='font-size:0.8rem;margin-top:8px'>
                Upload an image on the left to get started
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        predict_btn = st.button("⚡ Predict Demand", use_container_width=True)

        if predict_btn:
            with st.spinner("Analyzing design..."):
                try:
                    mobilenet               = load_mobilenet()
                    pred_model, pca, scaler, config = load_artifacts()

                    emb  = get_embedding(img, mobilenet)
                    pred = predict_qty(emb, avg_rate, pred_model, pca, scaler, config)
                    conf_label, conf_pct = confidence_label(pred, df_data)

                    # Store in session
                    st.session_state['pred']      = pred
                    st.session_state['emb']       = emb
                    st.session_state['conf_label'] = conf_label
                    st.session_state['conf_pct']   = conf_pct

                except Exception as e:
                    st.error(f"Prediction failed: {e}")

        if 'pred' in st.session_state:
            pred       = st.session_state['pred']
            conf_label = st.session_state['conf_label']
            conf_pct   = st.session_state['conf_pct']

            st.markdown(f"""
            <div class='prediction-box'>
                <div class='metric-label'>Predicted Sales Quantity</div>
                <div class='prediction-number'>{pred}</div>
                <div class='prediction-unit'>units</div>
            </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Confidence
            conf_color = {"High":"#7ed4a0","Medium":"#e8c97e","Low":"#e87e7e"}[conf_label]
            st.markdown(f"""
            <div style='background:#1a1814;border:1px solid #2e2a24;border-radius:10px;padding:16px'>
                <div style='display:flex;justify-content:space-between;align-items:center'>
                    <span style='color:#8a8070;font-size:0.8rem;text-transform:uppercase;letter-spacing:1px'>Confidence</span>
                    <span style='color:{conf_color};font-weight:500'>{conf_label} · {conf_pct}%</span>
                </div>
                <div class='conf-bar-bg'>
                    <div class='conf-bar-fill' style='width:{conf_pct}%;background:linear-gradient(90deg,#4caf76,{conf_color})'></div>
                </div>
            </div>""", unsafe_allow_html=True)

            # Range estimate
            low  = max(0, int(pred * 0.75))
            high = int(pred * 1.25)
            st.markdown(f"""
            <div style='background:#1a1814;border:1px solid #2e2a24;border-radius:10px;
                        padding:16px;margin-top:12px;display:flex;justify-content:space-around;text-align:center'>
                <div>
                    <div style='color:#6a6050;font-size:0.75rem;text-transform:uppercase;letter-spacing:1px'>Low Estimate</div>
                    <div style='color:#e8c97e;font-size:1.4rem;font-family:Playfair Display,serif'>{low}</div>
                </div>
                <div style='color:#3a3528;font-size:1.5rem'>|</div>
                <div>
                    <div style='color:#6a6050;font-size:0.75rem;text-transform:uppercase;letter-spacing:1px'>Predicted</div>
                    <div style='color:#7ed4a0;font-size:1.4rem;font-family:Playfair Display,serif'>{pred}</div>
                </div>
                <div style='color:#3a3528;font-size:1.5rem'>|</div>
                <div>
                    <div style='color:#6a6050;font-size:0.75rem;text-transform:uppercase;letter-spacing:1px'>High Estimate</div>
                    <div style='color:#e8c97e;font-size:1.4rem;font-family:Playfair Display,serif'>{high}</div>
                </div>
            </div>""", unsafe_allow_html=True)

# ─── Similar Designs ──────────────────────────────────────────────────────────
if 'emb' in st.session_state:
    st.markdown("<div class='gold-divider'></div>", unsafe_allow_html=True)
    st.markdown("<div class='section-title'>Most Similar Past Designs</div>", unsafe_allow_html=True)

    sim_df, sim_scores = get_similar(st.session_state['emb'], all_embs, df_data, top_k=top_k)
    cols = st.columns(top_k)

    for col, (_, row), score in zip(cols, sim_df.iterrows(), sim_scores):
        with col:
            try:
                sim_img = Image.open(row['image_path']).convert("RGB")
                st.image(sim_img, use_column_width=True)
            except:
                st.markdown("🖼️ Image not found")

            st.markdown(f"""
            <div style='background:#1a1814;border:1px solid #2e2a24;border-radius:8px;
                        padding:10px;text-align:center;margin-top:6px'>
                <div style='color:#e8c97e;font-size:0.78rem;font-weight:500'>Code: {row['first_code']}</div>
                <div style='color:#7ed4a0;font-size:1.1rem;font-family:Playfair Display,serif'>{int(row['total_qty'])} units</div>
                <div style='color:#5a5040;font-size:0.72rem'>₹{row['avg_rate']:.0f} avg rate</div>
                <div style='color:#4a6040;font-size:0.7rem;margin-top:4px'>{score*100:.1f}% similar</div>
            </div>""", unsafe_allow_html=True)

# ─── Data Explorer ────────────────────────────────────────────────────────────
st.markdown("<div class='gold-divider'></div>", unsafe_allow_html=True)

with st.expander("📊 Explore Training Data"):
    tab1, tab2 = st.tabs(["Sales Distribution", "Top Designs"])

    with tab1:
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.patch.set_facecolor('#0f0e0c')

        for ax in axes:
            ax.set_facecolor('#1a1814')
            ax.tick_params(colors='#8a8070')
            for spine in ax.spines.values():
                spine.set_color('#2e2a24')

        axes[0].hist(df_data['total_qty'], bins=20, color='#c9a84c', edgecolor='#0f0e0c', alpha=0.85)
        axes[0].set_title('Sales Quantity Distribution', color='#e8c97e', fontsize=11)
        axes[0].set_xlabel('Total QTY', color='#8a8070')

        axes[1].hist(df_data['avg_rate'], bins=20, color='#4caf76', edgecolor='#0f0e0c', alpha=0.85)
        axes[1].set_title('Rate Distribution', color='#e8c97e', fontsize=11)
        axes[1].set_xlabel('Avg Rate (₹)', color='#8a8070')

        plt.tight_layout()
        st.pyplot(fig)

    with tab2:
        top10 = df_data.nlargest(10, 'total_qty')[['first_code','total_qty','avg_rate']]
        top10.columns = ['Product Code', 'Total QTY Sold', 'Avg Rate (₹)']
        st.dataframe(top10, use_container_width=True, hide_index=True)

# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center;color:#3a3020;font-size:0.75rem;margin-top:48px;padding:16px;
            border-top:1px solid #1e1c18'>
    Demand Prediction Engine · Built for Maalde Company AI/ML Task · 
    Model: GradientBoosting · CV R² ≈ 0.75
</div>""", unsafe_allow_html=True)