import streamlit as st

st.set_page_config(layout="wide")

# --- Tesla background with styling and bottom corner credit ---
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://wallpapercave.com/wp/wp1821381.jpg");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
        position: relative;
    }

    .stApp::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        height: 100%;
        width: 100%;
        background-color: rgba(0, 0, 0, 0.8);  /* deeper dark overlay */
        z-index: -1;
    }

    .typewriter-subtitle {
        text-align: center;
        font-size: 2.2em;
        font-weight: bold;
        font-family: 'Courier New', monospace;
        color: #00ffff;
        text-shadow: 2px 2px 6px rgba(0, 255, 255, 0.8);
        white-space: nowrap;
        overflow: hidden;
        border-right: 0.15em solid #00ffff;
        margin-top: 60px;
        width: 26ch;
        animation: typing 4s steps(26) infinite alternate, blink-caret 0.75s step-end infinite;
    }

    @keyframes typing {
        from { width: 0 }
        to { width: 26ch }
    }

    @keyframes blink-caret {
        from, to { border-color: transparent }
        50% { border-color: #00ffff }
    }

    .bottom-credit {
        position: fixed;
        bottom: 10px;
        right: 20px;
        color: #bbbbbb;
        font-size: 0.9em;
        font-family: Arial, sans-serif;
        text-shadow: 1px 1px 3px black;
        z-index: 9999;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Animated subtitle ---
st.markdown('<div class="typewriter-subtitle">Powered by the legacy of Nikola Tesla</div>', unsafe_allow_html=True)

# --- Creator credit ---
st.markdown('<div class="bottom-credit">Created by NIRUPAM</div>', unsafe_allow_html=True)
