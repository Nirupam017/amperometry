import streamlit as st

st.set_page_config(layout="wide")

st.markdown("""
    <style>
    .stApp {
        background-image: url("https://wallpapercave.com/w/wp1936436");
        background-size: cover;
        background-position: center;
        color: white;
        text-align: center;
    }

    .overlay-text {
        margin-top: 20vh;
        font-size: 3em;
        font-weight: bold;
        color: #00ffff;
        text-shadow: 2px 2px 4px #000000;
    }

    .footer {
        position: fixed;
        bottom: 58px;  /* moved 1 cm up */
        right: 30px;
        font-size: 16px;
        font-weight: bold;
        background-color: rgba(0,0,0,0.6);
        padding: 6px 12px;
        border-radius: 8px;
        color: #00ffff;
        box-shadow: 0 0 10px #000;
        z-index: 9999;
    }
    </style>

    <div class="overlay-text">Inspired by the legacy of Nikola Tesla</div>
    <div class="footer">Created by <span style="color:#00ffff;">NIRUPAM</span></div>
""", unsafe_allow_html=True)
