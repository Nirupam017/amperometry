import streamlit as st
import base64

# Set wide layout
st.set_page_config(layout="wide")

# --- FUNCTION: Convert local image to base64 ---
def get_base64_image(path):
    with open(path, "rb") as image_file:
        encoded = base64.b64encode(image_file.read()).decode()
    return encoded

# --- LOAD and SET BACKGROUND ---
img_path = "D:/myWORK/PLOTS_AMPEROMETRY/TESLA.jpg"  # or TESLA.png
img_base64 = get_base64_image(img_path)

st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{img_base64}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    .stApp::before {{
        content: "";
        position: absolute;
        top: 0;
        left: 0;
        height: 100%;
        width: 100%;
        background-color: rgba(255, 255, 255, 0.4);  /* optional white overlay */
        z-index: -1;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# --- Title or Content Here ---
st.title("🔋 Nikola Tesla Streamlit App")
st.write("This is a minimal app with Nikola Tesla as the background.")
