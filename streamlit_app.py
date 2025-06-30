# streamlit_app.py

import streamlit as st

st.set_page_config(page_title="Biosensing Toolkit", layout="wide")

st.title("🧪 Biosensing Web App")
st.markdown("""
Welcome! This app has the following analysis tools:
- 📈 **Amperometry Plotting & Calibration**
- ⚡ **CV Overlay Viewer**
- 📊 **Scan Rate & Randles–Ševčík Analysis**

Use the sidebar to navigate between them.
""")
