import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
from gtts import gTTS
import tempfile

st.set_page_config(
    page_title="DrawMate",
    page_icon="🎨",
    layout="wide"
)

st.title("🎨 DrawMate")
st.subheader("AI Drawing Assistant for Children")
st.write("Draw a House, Tree, or Sun and let DrawMate guide you! ✏️")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("DrawMate_final_model.keras")

model = load_model()

categories = ["house", "tree", "sun", "car", "flower"]

guidance = {
    "English": {
        "house": "Great job! You drew a house. First, draw a square for the main part. Next, add a triangle for the roof. Then add a door and two windows. Keep drawing and be creative!",
        "tree": "Wonderful! You drew a tree. First, draw the trunk. Next, add branches. Then draw the leafy crown. You can add grass, flowers, or birds!",
        "sun": "Awesome! You drew the sun. First, draw a circle. Next, add rays around it. You can add clouds, mountains, or a beautiful landscape!"
    },

    "Tamil": {
        "house": "மிகவும் நன்றாக வரைந்துள்ளீர்கள்! நீங்கள் ஒரு வீட்டை வரைந்துள்ளீர்கள். முதலில் வீட்டின் முக்கிய பகுதியை ஒரு சதுரமாக வரையுங்கள். அடுத்து கூரைக்கு ஒரு முக்கோணம் வரையுங்கள். பின்னர் கதவு மற்றும் இரண்டு ஜன்னல்களைச் சேருங்கள். தொடர்ந்து வரைந்து உங்கள் படைப்பாற்றலை வெளிப்படுத்துங்கள்!",
        "tree": "அருமை! நீங்கள் ஒரு மரத்தை வரைந்துள்ளீர்கள். முதலில் மரத்தின் தண்டை வரையுங்கள். அடுத்து கிளைகளை வரையுங்கள். பின்னர் இலைகளை வரையுங்கள். புல், பூக்கள் அல்லது பறவைகளையும் சேர்க்கலாம்!",
        "sun": "சிறப்பாக வரைந்துள்ளீர்கள்! நீங்கள் சூரியனை வரைந்துள்ளீர்கள். முதலில் ஒரு வட்டத்தை வரையுங்கள். அடுத்து அதைச் சுற்றி கதிர்களை வரையுங்கள். மேகங்கள், மலைகள் அல்லது அழகான இயற்கைக் காட்சியையும் சேர்க்கலாம்!"
    },

    "Malayalam": {
        "house": "വളരെ നന്നായി വരച്ചിരിക്കുന്നു! നിങ്ങൾ ഒരു വീട് വരച്ചിരിക്കുന്നു. ആദ്യം വീടിന്റെ പ്രധാന ഭാഗം ഒരു ചതുരമായി വരയ്ക്കുക. തുടർന്ന് മേൽക്കൂരയ്ക്കായി ഒരു ത്രികോണം വരയ്ക്കുക. പിന്നീട് ഒരു വാതിലും രണ്ട് ജനലുകളും ചേർക്കുക. തുടർന്നും വരച്ച് നിങ്ങളുടെ സർഗ്ഗാത്മകത പ്രകടിപ്പിക്കുക!",
        "tree": "വളരെ മനോഹരം! നിങ്ങൾ ഒരു മരം വരച്ചിരിക്കുന്നു. ആദ്യം മരത്തിന്റെ തണ്ട് വരയ്ക്കുക. തുടർന്ന് ശാഖകൾ വരയ്ക്കുക. പിന്നീട് ഇലകൾ വരയ്ക്ക."},
