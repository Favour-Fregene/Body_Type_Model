import streamlit as st
import numpy as np
import joblib

st.set_page_config(
    page_title='African Women Body Type & Outfit Recommender',
    page_icon='👗',
    layout='centered'
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700&family=Lato:wght@300;400;700&display=swap');
    html, body, [class*="css"] { font-family: 'Lato', sans-serif; }

    .stApp {
        background-image: url('https://images.unsplash.com/photo-1625646741211-711bdd65c570?fm=jpg&q=60&w=1600&auto=format&fit=crop');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }

    .stButton > button {
        background-color: #722F37;
        color: white;
        font-family: 'Playfair Display', serif;
        font-size: 1.05em;
        letter-spacing: 1px;
        border: none;
        padding: 14px;
        border-radius: 8px;
        width: 100%;
    }
    .stButton > button:hover { background-color: #8B0000; }
    .stNumberInput label { color: #FFFFFF !important; font-weight: 700; font-size: 1em; }
</style>
""", unsafe_allow_html=True)

# Load model
best_svm = joblib.load('svm_model.pkl')
scaler   = joblib.load('scaler.pkl')
le       = joblib.load('label_encoder.pkl')

OUTFITS = {
    'Hourglass': {
        'description': 'Balanced bust and hip with a defined waist.',
        'goal'       : 'Highlight your natural waist and celebrate your curves.',
        'best_fits'  : ['Wrap dresses', 'Belted outfits', 'Fit-and-flare styles', 'Bodycon dresses', 'High-waisted trousers', 'Ankara wrap skirts'],
        'avoid'      : ['Boxy oversized styles', 'Drop-waist designs'],
        'neckline'   : 'V-neck, sweetheart, scoop',
        'fabric'     : 'Soft draping fabrics like jersey, silk, chiffon',
        'african'    : 'Ankara wrap dress or off-shoulder Iro and Buba'
    },
    'Pear': {
        'description': 'Hips wider than bust — very common African female silhouette.',
        'goal'       : 'Balance upper and lower body by drawing attention upward.',
        'best_fits'  : ['A-line skirts', 'Peplum tops', 'Off-shoulder tops', 'Wide-leg trousers', 'Ankara peplum tops'],
        'avoid'      : ['Tight pencil skirts', 'Bold patterns on lower half'],
        'neckline'   : 'Boat neck, off-shoulder, square neck',
        'fabric'     : 'Structured fabrics on top, flowing on bottom',
        'african'    : 'Peplum Ankara blouse with a plain dark wrapper skirt'
    },
    'Apple': {
        'description': 'Fuller midsection with slimmer hips and legs.',
        'goal'       : 'Draw attention away from midsection and elongate the torso.',
        'best_fits'  : ['Empire waist dresses', 'Flowy tunics', 'A-line tops', 'Structured blazers', 'Kaftan styles'],
        'avoid'      : ['Belted styles at the waist', 'Clingy fabrics', 'Cropped tops'],
        'neckline'   : 'V-neck, deep scoop',
        'fabric'     : 'Flowy draping fabrics that skim without clinging',
        'african'    : 'Kaftan or Boubou in bold Ankara or Adire print'
    },
    'Rectangle': {
        'description': 'Bust, waist, and hip measurements are similar.',
        'goal'       : 'Create the illusion of curves and a defined waist.',
        'best_fits'  : ['Ruffled tops', 'Peplum styles', 'Belted outfits', 'Fit-and-flare dresses', 'Ankara co-ord sets with a belt'],
        'avoid'      : ['Boxy shapeless styles', 'Drop-waist cuts'],
        'neckline'   : 'Any neckline — embellished or statement works best',
        'fabric'     : 'Structured fabrics that add shape',
        'african'    : 'Structured Aso-Oke or Ankara co-ord set with a bold belt'
    },
    'Inverted Triangle': {
        'description': 'Shoulders and bust wider than hips.',
        'goal'       : 'Balance broad shoulders by adding volume to lower body.',
        'best_fits'  : ['A-line and flared skirts', 'Wide-leg trousers', 'Scoop neck tops', 'Full skirts with plain fitted tops'],
        'avoid'      : ['Off-shoulder tops', 'Shoulder pads', 'Slim straight-leg trousers'],
        'neckline'   : 'Scoop neck, round neck, V-neck',
        'fabric'     : 'Plain minimal fabric on top, bold textured fabric on bottom',
        'african'    : 'Plain fitted top with a voluminous Ankara bubble skirt'
    }
}

def predict_and_recommend(bust, waist, hip, height):
    whr      = waist / hip
    bhr      = bust  / hip
    wbr      = waist / bust
    features = scaler.transform([[bust, waist, hip, height, whr, bhr, wbr]])
    pred      = best_svm.predict(features)[0]
    body_type = le.inverse_transform([pred])[0]
    return body_type, OUTFITS[body_type]

# ── HEADER ──
st.markdown("""
<div style='background:rgba(10,25,80,0.95); padding:28px; border-radius:14px;
            margin-bottom:24px; border:2px solid #722F37;'>
    <p style='font-family:Playfair Display,serif; color:#FFFFFF; text-align:center;
              font-size:2.2em; font-weight:700; letter-spacing:2px; margin:0;'>
        👗 African Women Body Type & Outfit Recommender
    </p>
    <p style='color:#B0C4DE; text-align:center; font-size:1em;
              margin-top:8px; margin-bottom:0;'>
        Enter your measurements below to discover your body type
        and get personalised outfit recommendations
    </p>
</div>
""", unsafe_allow_html=True)

# ── INPUT BOX ──
st.markdown("""
<div style='background:rgba(10,25,80,0.92); padding:20px 24px 8px;
            border-radius:12px; border:1px solid #722F37; margin-bottom:12px;'>
    <p style='font-family:Playfair Display,serif; color:#E8C4C4; font-size:1.1em;
              border-bottom:1px solid #722F37; padding-bottom:6px; margin-bottom:12px;'>
        📏 Your Measurements (in centimetres)
    </p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    bust   = st.number_input('Bust (cm)',   min_value=60.0,  max_value=140.0, value=96.0,  step=0.5)
    waist  = st.number_input('Waist (cm)',  min_value=50.0,  max_value=120.0, value=72.0,  step=0.5)
with col2:
    hip    = st.number_input('Hip (cm)',    min_value=65.0,  max_value=145.0, value=102.0, step=0.5)
    height = st.number_input('Height (cm)', min_value=140.0, max_value=200.0, value=162.0, step=0.5)

st.markdown("<br>", unsafe_allow_html=True)

if st.button('✨ Discover My Body Type & Outfit Recommendations'):

    body_type, rec = predict_and_recommend(bust, waist, hip, height)

    # ── BODY TYPE — Blue Box ──
    st.markdown(f"""
    <div style='background:rgba(10,25,80,0.97); padding:24px; border-radius:14px;
                border:2px solid #1a3a8f; text-align:center; margin-bottom:16px;'>
        <p style='font-family:Playfair Display,serif; color:#B0C4DE;
                  font-size:1em; letter-spacing:3px; margin:0;'>YOUR BODY TYPE IS</p>
        <p style='font-family:Playfair Display,serif; color:#FFFFFF;
                  font-size:2.8em; font-weight:700; letter-spacing:4px; margin:6px 0;'>
            {body_type.upper()}
        </p>
        <p style='color:#D6E4F0; font-style:italic; margin:0;'>{rec['description']}</p>
    </div>
    """, unsafe_allow_html=True)

    # ── MEASUREMENTS BOX ──
    st.markdown(f"""
    <div style='background:rgba(10,25,80,0.92); padding:16px 20px; border-radius:10px;
                border:1px solid #722F37; margin-bottom:16px; text-align:center;'>
        <p style='font-family:Playfair Display,serif; color:#E8C4C4;
                  font-size:1em; margin:0 0 8px; font-weight:700;'>📐 Your Measurements</p>
        <p style='color:#FFFFFF; font-size:1.05em; font-weight:700; margin:0;'>
            Bust: {bust}cm &nbsp;|&nbsp; Waist: {waist}cm &nbsp;|&nbsp;
            Hip: {hip}cm &nbsp;|&nbsp; Height: {height}cm
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── STYLING GOAL BOX ──
    st.markdown(f"""
    <div style='background:rgba(10,25,80,0.92); padding:14px 18px; border-radius:10px;
                border:1px solid #722F37; margin-bottom:16px;
                border-left:4px solid #722F37;'>
        <p style='color:#FFFFFF; margin:0; font-size:1em; font-weight:700;'>
            🎯 Styling Goal: <span style='font-weight:400;'>{rec['goal']}</span>
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── BEST FITS & AVOID — In One Box ──
    fits_html = ''.join(f"<p style='color:#FFFFFF; font-size:1em; font-weight:700; margin:4px 0;'>• {f}</p>" for f in rec['best_fits'])
    avoid_html = ''.join(f"<p style='color:#FFB3B3; font-size:1em; font-weight:700; margin:4px 0;'>• {a}</p>" for a in rec['avoid'])

    st.markdown(f"""
    <div style='background:rgba(10,25,80,0.92); padding:18px 20px; border-radius:10px;
                border:1px solid #722F37; margin-bottom:16px;'>
        <div style='display:grid; grid-template-columns:1fr 1fr; gap:20px;'>
            <div>
                <p style='font-family:Playfair Display,serif; color:#E8C4C4;
                          font-size:1.05em; font-weight:700; border-bottom:1px solid #722F37;
                          padding-bottom:6px; margin-bottom:10px;'>✅ Best Fits</p>
                {fits_html}
            </div>
            <div>
                <p style='font-family:Playfair Display,serif; color:#E8C4C4;
                          font-size:1.05em; font-weight:700; border-bottom:1px solid #722F37;
                          padding-bottom:6px; margin-bottom:10px;'>❌ Avoid</p>
                {avoid_html}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── NECKLINE & FABRIC — In One Box ──
    st.markdown(f"""
    <div style='background:rgba(10,25,80,0.92); padding:18px 20px; border-radius:10px;
                border:1px solid #722F37; margin-bottom:16px;'>
        <div style='display:grid; grid-template-columns:1fr 1fr; gap:20px;'>
            <div>
                <p style='color:#E8C4C4; font-size:1.05em; font-weight:700;
                          margin:0 0 6px;'>👗 Neckline</p>
                <p style='color:#FFFFFF; font-size:1em; font-weight:700;
                          margin:0;'>{rec['neckline']}</p>
            </div>
            <div>
                <p style='color:#E8C4C4; font-size:1.05em; font-weight:700;
                          margin:0 0 6px;'>🧵 Fabric Tip</p>
                <p style='color:#FFFFFF; font-size:1em; font-weight:700;
                          margin:0;'>{rec['fabric']}</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── AFRICAN STYLE — Normal Size ──
    st.markdown(f"""
    <div style='background:#722F37; padding:18px 22px; border-radius:12px;
                margin-top:4px;'>
        <p style='color:#FFE4E1; font-family:Playfair Display,serif;
                  font-size:1em; font-weight:700; margin:0 0 6px; letter-spacing:1px;'>
            🌍 African Style Recommendation
        </p>
        <p style='color:#FFFFFF; font-size:1em; font-weight:700; margin:0;'>
            {rec['african']}
        </p>
    </div>
    """, unsafe_allow_html=True)