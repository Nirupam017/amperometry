import streamlit as st

# Set layout
st.set_page_config(layout="wide")

# --- Apply Tesla image from online URL as background ---
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
        background-color: rgba(255, 255, 255, 0.3); /* optional white overlay for readability */
        z-index: -1;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- App content ---
st.title("⚡ Amperometry Analysis")
st.write("This app is powered by the brilliance of Nikola Tesla.")
