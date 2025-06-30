import streamlit as st

st.set_page_config(layout="wide")

st.markdown("""
<style>
.stApp {
  background-image: url("https://wallpapercave.com/wp/wp1821381.jpg");
  background-size: cover;
  background-position: center;
}
.stApp::before {
  content: "";
  position: absolute;
  inset: 0;
  background-color: rgba(0,0,0,0.75);
  z-index: -1;
}
.typing-text {
  font-family: monospace;
  font-size: 2em;
  color: #00ffff;
  white-space: nowrap;
  overflow: hidden;
  border-right: 0.15em solid #00ffff;
  width: 26ch;
  margin: 100px auto;
  animation: typing 4s steps(26, end) infinite alternate, blink-caret .7s step-end infinite;
  text-align: center;
}
@keyframes typing { from { width: 0 } to { width: 26ch } }
@keyframes blink-caret { from, to { border-color: transparent } 50% { border-color: #00ffff } }
.footer-credit {
  position:fixed; bottom:15px; right:25px;
  color:#bbb; text-shadow:1px 1px 3px black;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="typing-text">Powered by the legacy of Nikola Tesla</div>', unsafe_allow_html=True)
st.markdown('<div class="footer-credit">Created by NIRUPAM</div>', unsafe_allow_html=True)
