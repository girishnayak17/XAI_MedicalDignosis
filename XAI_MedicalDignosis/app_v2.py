"""Multi-page XAI Chest X-ray application entry point."""

import streamlit as st

st.set_page_config(
    page_title="XAI Chest X-Ray",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Dark visual system and a compact horizontal page navigation.
st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] { background: #0b1220; }
    [data-testid="stHeader"] { background: rgba(11,18,32,0.92); }
    [data-testid="stSidebar"] { background: #111c2e; }
    .block-container { padding-top: 1.25rem; padding-bottom: 2rem; max-width: 1440px; }
    [data-testid="stPageLink-NavLink"] { border-radius: 10px; padding: .45rem .7rem; }
    [data-testid="stPageLink-NavLink"]:hover { background: #1b2b43; }
    div[data-testid="stMetric"] { background: #111c2e; border: 1px solid #26364d; padding: 14px; border-radius: 12px; }
    div[data-testid="stFileUploader"] { background: #111c2e; border: 1px dashed #40536d; border-radius: 12px; padding: 12px; }
    .xai-subtitle { color: #a9bad0; margin-top: -10px; }
    </style>
    """,
    unsafe_allow_html=True,
)

from src.ui.core import initialize_session_state

initialize_session_state()

st.markdown("# 🩻  XAI Chest X-Ray")
st.markdown(
    '<p class="xai-subtitle">Explainable AI for four-class chest X-ray classification</p>',
    unsafe_allow_html=True,
)

pages = [
    st.Page("pages/home.py", title="Home", icon="🏠", default=True),
    st.Page("pages/xray_analysis.py", title="X-Ray Analysis", icon="🩻"),
    st.Page("pages/eda.py", title="EDA", icon="📊"),
    st.Page("pages/model_performance.py", title="Model Performance", icon="📈"),
    st.Page("pages/gradcam.py", title="Grad-CAM", icon="🔍"),
    st.Page("pages/shap.py", title="SHAP", icon="🧠"),
    st.Page("pages/ai_explanation.py", title="AI Explanation", icon="✨"),
    st.Page("pages/about.py", title="About & User Guide", icon="ℹ️"),
]

# Hide Streamlit's default navigation and render our own horizontal navbar.
navigation = st.navigation(pages, position="hidden")
nav_items = [
    ("Home", "pages/home.py", "🏠"),
    ("Analysis", "pages/xray_analysis.py", "🩻"),
    ("EDA", "pages/eda.py", "📊"),
    ("Models", "pages/model_performance.py", "📈"),
    ("Grad-CAM", "pages/gradcam.py", "🔍"),
    ("SHAP", "pages/shap.py", "🧠"),
    ("AI", "pages/ai_explanation.py", "✨"),
    ("About", "pages/about.py", "ℹ️"),
]
nav_columns = st.columns(len(nav_items))
for column, (label, page_path, icon) in zip(nav_columns, nav_items):
    with column:
        st.page_link(page_path, label=label, icon=icon, use_container_width=True)

st.divider()
navigation.run()
