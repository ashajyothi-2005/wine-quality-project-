import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import pickle
import base64
from pathlib import Path

st.set_page_config(page_title="Wine Quality Expert", layout="centered")

APP_DIR = Path(__file__).parent

# ---------- Helpers ----------
def file_to_base64(path: Path) -> str:
    if path.exists():
        data = path.read_bytes()
        return base64.b64encode(data).decode("utf-8")
    return ""

def inject_css(bg_image_data_uri: str | None, bg_image_url: str | None):
    bg_css = f"background-image: url('{bg_image_data_uri}');" if bg_image_data_uri else f"background-image: url('{bg_image_url}');"
    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=Lato:wght@300;400;500;700&display=swap');
.stApp {{ {bg_css} background-size: cover; background-position: center; background-attachment: fixed; }}
.stApp:before {{ content: ""; position: fixed; inset: 0; background: linear-gradient(180deg, rgba(10, 6, 8, 0.72) 0%, rgba(10, 6, 8, 0.55) 45%, rgba(10, 6, 8, 0.78) 100%); z-index: 0; }}
[data-testid="block-container"] {{ position: relative; z-index: 1; max-width: 980px; }}

/* --- ANIMATIONS --- */
@keyframes bottle-fall {{
    0% {{ transform: translateY(-20vh) rotate(0deg); opacity: 0; }}
    10% {{ opacity: 1; }}
    90% {{ opacity: 1; }}
    100% {{ transform: translateY(100vh) rotate(720deg); opacity: 0; }}
}}

@keyframes snow-fall {{
    0% {{ transform: translateY(-10vh) translateX(0); opacity: 0; }}
    10% {{ opacity: 0.8; }}
    100% {{ transform: translateY(100vh) translateX(20px); opacity: 0; }}
}}

.falling-bottle {{
    position: fixed;
    top: -50px;
    font-size: 45px; 
    z-index: 9999;
    animation: bottle-fall linear infinite;
    pointer-events: none;
    user-select: none;
}}

.snow-flake {{
    position: fixed;
    top: -10px;
    background: white;
    border-radius: 50%;
    z-index: 9998;
    opacity: 0.6;
    animation: snow-fall linear infinite;
    pointer-events: none;
}}

.wine-title {{ font-family: "Playfair Display", serif; font-weight: 700; font-size: 3.1rem; text-align: center; color: #f5f0e6; margin-bottom: 0.4rem; }}
.wine-title .gold {{ color: #d4af37; }}
.wine-subtitle {{ font-family: "Lato", sans-serif; font-size: 1.12rem; text-align: center; color: rgba(245, 240, 230, 0.72); margin-bottom: 1.7rem; }}

/* Form Inputs */
label {{ color: #f5f0e6 !important; font-family: "Lato", sans-serif !important; font-weight: 600 !important; }}
.stNumberInput input {{ background: rgba(68, 28, 40, 0.62) !important; border: 1px solid rgba(212, 175, 55, 0.2) !important; color: white !important; border-radius: 14px !important; }}
.stButton > button {{ width: 100%; border-radius: 16px !important; padding: 16px !important; font-weight: 700 !important; background: linear-gradient(135deg, #380a16 0%, #781a34 100%) !important; color: white !important; border: none !important; }}
</style>
""", unsafe_allow_html=True)

def play_wine_sound():
    sound_url = "https://www.soundjay.com/misc/pouring-liquid-1.mp3"
    st.markdown(f"""
        <audio autoplay="true">
            <source src="{sound_url}" type="audio/mpeg">
        </audio>
        """, unsafe_allow_html=True)

# --- App Logic ---
local_bg = APP_DIR / "wine-cellar-bg.jpg"
bg_uri = f"data:image/jpeg;base64,{file_to_base64(local_bg)}" if local_bg.exists() else None
inject_css(bg_uri, "https://images.unsplash.com/photo-1506377247377-2a5b3b417ebb?q=80&w=1920")

try:
    scaler = pickle.load(open(APP_DIR / "scaler_model.sav", "rb"))
    rf_model = pickle.load(open(APP_DIR / "finalized_RFmodel.sav", "rb"))
except:
    st.error("Model files missing.")
    st.stop()

st.markdown('<div class="wine-title">🍷 Wine Quality <span class="gold">Expert</span> ✨</div>', unsafe_allow_html=True)
st.markdown('<div class="wine-subtitle">Fine-tune chemical properties and predict quality score instantly.</div>', unsafe_allow_html=True)

with st.form("wine_form"):
    col1, col2 = st.columns(2)
    with col1:
        fa = st.number_input("Fixed Acidity", 7.4)
        va = st.number_input("Volatile Acidity", 0.7)
        ca = st.number_input("Citric Acid", 0.0)
        rs = st.number_input("Residual Sugar", 1.9)
        ch = st.number_input("Chlorides", 0.076, format="%.4f")
    with col2:
        fsd = st.number_input("Free Sulfur Dioxide", 11.0)
        tsd = st.number_input("Total Sulfur Dioxide", 34.0)
        de = st.number_input("Density", 0.9978, format="%.4f")
        ph = st.number_input("pH", 3.51)
        su = st.number_input("Sulphates", 0.56)
        al = st.number_input("Alcohol", 9.4)
    submitted = st.form_submit_button("🔍 ANALYZE WINE SAMPLE")

if submitted:
    # 1. Prediction Logic
    input_data = pd.DataFrame([[fa, va, ca, np.log(rs+1e-6), np.log(ch), np.log(fsd), np.log(tsd), de, ph, np.log(su), al]], 
                               columns=["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"])
    
    pred = float(rf_model.predict(scaler.transform(input_data))[0])
    
    # 2. 5-Tier Classification Label
    if pred >= 8.0:
        label = "BEST QUALITY"
    elif pred >= 7.0:
        label = "BETTER QUALITY"
    elif pred >= 6.0:
        label = "GOOD QUALITY"
    elif pred >= 5.0:
        label = "AVERAGE QUALITY"
    else:
        label = "BASIC QUALITY"
    
    play_wine_sound()

    # 3. Snowfall Generation
    snow_html = "".join([f'<div class="snow-flake" style="left: {np.random.randint(0,100)}%; width: {np.random.randint(3,10)}px; height: {np.random.randint(3,10)}px; animation-duration: {np.random.uniform(3,8)}s; animation-delay: {np.random.uniform(0,4)}s;"></div>' for _ in range(80)])

    # 4. Result Display
    st.markdown(f"""
        {snow_html}
        <div class="falling-bottle" style="left: 3%; animation-delay: 0s; animation-duration: 4s;">🍾</div>
        <div class="falling-bottle" style="left: 15%; animation-delay: 2.1s; animation-duration: 6s;">🍷</div>
        <div class="falling-bottle" style="left: 30%; animation-delay: 1.5s; animation-duration: 5s;">🍾</div>
        <div class="falling-bottle" style="left: 45%; animation-delay: 3.5s; animation-duration: 7s;">🍷</div>
        <div class="falling-bottle" style="left: 60%; animation-delay: 0.8s; animation-duration: 4.5s;">🍾</div>
        <div class="falling-bottle" style="left: 75%; animation-delay: 2.8s; animation-duration: 5.5s;">🍷</div>
        <div class="falling-bottle" style="left: 90%; animation-delay: 1.2s; animation-duration: 6.5s;">🍾</div>
        
        <div style="position: relative; margin: 30px auto; max-width: 650px; z-index: 10;">
            <div style="background: rgba(20, 20, 20, 0.45); backdrop-filter: blur(15px); -webkit-backdrop-filter: blur(15px); border: 4px solid #f1c40f; border-radius: 40px; padding: 60px 20px; text-align: center; box-shadow: 0 10px 40px rgba(0,0,0,0.8);">
                <h4 style="color: #f1c40f; font-family: 'Lato', sans-serif; letter-spacing: 5px; font-size: 1rem; margin-bottom: 10px; text-transform: uppercase;">PREDICTED SCORE</h4>
                <div style="font-family: 'Playfair Display', serif; font-size: 9rem; font-weight: 700; color: #ffffff; line-height: 1; margin: 10px 0;">{pred:.1f}</div>
                <div style="display: inline-block; margin-top: 20px; padding: 12px 50px; border: 3px solid #f1c40f; border-radius: 50px; color: #f1c40f; font-family: 'Lato', sans-serif; font-weight: 700; letter-spacing: 3px; text-transform: uppercase; font-size: 1.1rem;">
                    {label}
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

st.markdown('<div style="margin-top: 40px; text-align: center; font-family: Lato, sans-serif; font-size: 0.86rem; color: rgba(245, 240, 230, 0.45);">Powered by Machine Learning • Premium Wine Analysis</div>', unsafe_allow_html=True)