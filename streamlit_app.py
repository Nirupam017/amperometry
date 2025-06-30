import streamlit as st

st.set_page_config(layout="wide")

# --- Tesla background with dark overlay and enhanced text style ---
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
        background-color: rgba(0, 0, 0, 0.75);  /* stronger dark overlay */
        z-index: -1;
    }

    .typewriter-subtitle {
        text-align: center;
        font-size: 2.4em;
        font-weight: bold;
        font-family: 'Courier New', monospace;
        color: #00ccff;
        text-shadow: 2px 2px 6px rgba(0, 0, 0, 0.9); /* glow effect */
        white-space: nowrap;
        overflow: hidden;
        border-right: 0.15em solid #00ccff;
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
        50% { border-color: #00ccff; }
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Only this line is shown in animated glowing text ---
st.markdown('<div class="typewriter-subtitle">Powered by the legacy of Nikola Tesla</div>', unsafe_allow_html=True)

