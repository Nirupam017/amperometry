import streamlit as st

st.set_page_config(layout="wide")

# Custom CSS for background, animation, and footer
st.markdown("""
<style>
/* Background Tesla image with overlay */
.stApp {
    background-image: url("https://wallpapercave.com/wp/wp1821381.jpg");
    background-size: cover;
    background-attachment: fixed;
    background-position: center;
    background-repeat: no-repeat;
    position: relative;
}
.stApp::before {
    content: "";
    position: absolute;
    inset: 0;
    background-color: rgba(0, 0, 0, 0.75); /* dark overlay */
    z-index: -1;
}

/* Typewriter effect for subtitle */
.typing-text {
    font-family: 'Courier New', monospace;
    font-size: 2.2em;
    font-weight: bold;
    color: #00ffff;
    text-shadow: 0 0 10px #00ffff;
    white-space: nowrap;
    overflow: hidden;
    border-right: 0.15em solid #00ffff;
    width: 26ch;
    margin: 100px auto 40px auto;
    animation: typing 4s steps(26, end) infinite alternate, blink-caret 0.75s step-end infinite;
    text-align: center;
}

@keyframes typing {
    from { width: 0 }
    to { width: 26ch }
}

@keyframes blink-caret {
    from, to { border-color: transparent }
    50% { border-color: #00ffff }
}

/* Bottom right credit */
.footer-credit {
    position: fixed;
    bottom: 15px;
    right: 25px;
    font-size: 1em;
    font-weight: bold;
    color: #bbbbbb;
    text-shadow: 1px 1px 4px black;
    z-index: 9999;
    font-family: Arial, sans-serif;
}
</style>
""", unsafe_allow_html=True)

# Subtitle animation text
st.markdown('<div class="typing-text">Powered by the legacy of Nikola Tesla</div>', unsafe_allow_html=True)

# Footer credit
st.markdown('<div class="footer-credit">Created by NIRUPAM</div>', unsafe_allow_html=True)
