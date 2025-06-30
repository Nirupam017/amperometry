import streamlit as st

st.set_page_config(layout="wide")

# Use custom CSS for background and text
st.markdown("""
    <style>
    .stApp {
        background-image: url("https://wallpapercave.com/wp/wp1821381.jpg");
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
        bottom: 20px;
        right: 30px;
        font-size: 16px;
        color: #ffffffcc;
        font-weight: bold;
        text-shadow: 1px 1px 3px black;
    }
    </style>

    <div class="overlay-text">Powered by the legacy of Nikola Tesla</div>
    <div class="footer">Created by NIRUPAM</div>
""", unsafe_allow_html=True)
