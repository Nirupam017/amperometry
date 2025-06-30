import streamlit as st

st.set_page_config(layout="wide")

# CSS and HTML
st.markdown("""
<style>
/* Full background image */
.stApp {
  background-image: url("https://wallpapercave.com/wp/wp1821381.jpg");
  background-size: cover;
  background-position: center;
  position: relative;
  color: white;
}

/* Add dark overlay */
.stApp::before {
  content: "";
  position: absolute;
  top: 0; left: 0; right: 0; bottom: 0;
  background-color: rgba(0,0,0,0.75);
  z-index: -1;
}

/* Typing effect */
.typing-container {
  text-align: center;
  margin-top: 20vh;
}

.typing-text {
  font-family: 'Courier New', Courier, monospace;
  font-size: 2.5em;
  font-weight: bold;
  color: #00FFFF;
  width: 28ch;
  white-space: nowrap;
  overflow: hidden;
  border-right: 4px solid #00FFFF;
  animation: typing 5s steps(28, end) infinite alternate,
             blink 0.7s step-end infinite;
  margin: auto;
}

@keyframes typing {
  from { width: 0; }
  to { width: 28ch; }
}

@keyframes blink {
  0%, 100% { border-color: transparent; }
  50% { border-color: #00FFFF; }
}

/* Footer credit */
.footer {
  position: fixed;
  bottom: 20px;
  right: 30px;
  color: #ffffffcc;
  font-size: 14px;
  font-weight: bold;
  text-shadow: 1px 1px 4px black;
  z-index: 10;
}
</style>

<div class="typing-container">
  <div class="typing-text">Powered by the legacy of Nikola Tesla</div>
</div>
<div class="footer">Created by NIRUPAM</div>
""", unsafe_allow_html=True)
