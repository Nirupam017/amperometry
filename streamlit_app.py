import streamlit as st

st.set_page_config(layout="wide")

# --- Tesla image as background with dark overlay ---
st.markdown(
    """
    <style>
    .stApp {
        background-image: url("https://wallpapercave.com/wp/wp1821381.jpg");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }

    .stApp::before {
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        height: 100%;
        width: 100%;
        background-color: rgba(0, 0, 0, 0.65); /* darkens the background for contrast */
        z-index: -1;
    }

    .typewriter-subtitle {
        text-align: center;
        color: #00ffee;
        font-size: 2.2em;
        font-family: 'Courier New', monospace;
        white-space: nowrap;
        overflow: hidden;
        border-right: 0.15em solid orange;
        margin-top: 50px;
        width: 100%;
        animation: typing 4s steps(40, end), blink-caret 0.75s step-end infinite;
    }

    @keyframes typing {
        from { width: 0; }
        to { width: 100%; }
    }

    @keyframes blink-caret {
        from, to { border-color: transparent; }
        50% { border-color: orange; }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Only this animated line appears ---
st.markdown('<div class="typewriter-subtitle">Powered by the legacy of Nikola Tesla</div>', unsafe_allow_html=True)
