import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
from gtts import gTTS
from streamlit_drawable_canvas import st_canvas
import tempfile


# ==============================
# PAGE SETTINGS
# ==============================

st.set_page_config(
    page_title="DrawMate",
    page_icon="🎨",
    layout="centered"
)

st.title("🎨 DrawMate")
st.subheader("AI-Powered Smart Drawing Assistant")

st.write(
    "Draw a picture directly below and let DrawMate "
    "understand your drawing! ✏️"
)


# ==============================
# LOAD MODEL
# ==============================

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "DrawMate_final_model.keras"
    )


try:
    model = load_model()
except Exception as e:
    st.error("❌ Could not load the AI model.")
    st.code(str(e))
    st.stop()


# ==============================
# MODEL CLASSES
# ==============================

categories = [
    "house",
    "tree",
    "sun",
    "car",
    "flower"
]


# ==============================
# LANGUAGE
# ==============================

language = st.selectbox(
    "🌐 Choose your language",
    [
        "English",
        "Tamil",
        "Malayalam",
        "Hindi"
    ]
)


# ==============================
# GUIDANCE
# ==============================

feedback = {

    "English": {

        "house":
        "Great job! You drew a house! "
        "First, draw a square or rectangle for the main part of the house. "
        "Next, add a triangle for the roof. "
        "Then add a door and windows. "
        "Finally, add trees, flowers or a garden.",

        "tree":
        "Wonderful! You drew a tree! "
        "First, draw the trunk. "
        "Next, add branches. "
        "Then draw the leaves. "
        "Finally, add grass, flowers or birds.",

        "sun":
        "Awesome! You drew the sun! "
        "First, draw a circle. "
        "Next, add rays around it. "
        "Make the rays different lengths. "
        "Finally, add clouds, mountains or a landscape."
    },

    "Tamil": {

        "house":
        "மிகவும் நன்றாக வரைந்துள்ளீர்கள்! நீங்கள் ஒரு வீட்டை வரைந்துள்ளீர்கள். "
        "முதலில் வீட்டின் முக்கிய பகுதியை சதுரம் அல்லது செவ்வகமாக வரையுங்கள். "
        "அடுத்து கூரையை முக்கோணமாக வரையுங்கள். "
        "பின்னர் கதவு மற்றும் ஜன்னல்களை சேர்க்குங்கள். "
        "இறுதியாக மரங்கள், பூக்கள் அல்லது தோட்டத்தை சேர்க்கலாம்.",

        "tree":
        "மிகவும் அருமை! நீங்கள் ஒரு மரத்தை வரைந்துள்ளீர்கள். "
        "முதலில் மரத்தின் தண்டை வரையுங்கள். "
        "அடுத்து கிளைகளை வரையுங்கள். "
        "பின்னர் இலைகளை வரையுங்கள். "
        "இறுதியாக புல், பூக்கள் அல்லது பறவைகளை சேர்க்கலாம்.",

        "sun":
        "அருமை! நீங்கள் சூரியனை வரைந்துள்ளீர்கள். "
        "முதலில் ஒரு வட்டத்தை வரையுங்கள். "
        "அடுத்து அதைச் சுற்றி கதிர்களை வரையுங்கள். "
        "கதிர்களை வெவ்வேறு நீளங்களில் வரையுங்கள். "
        "இறுதியாக மேகங்கள், மலைகள் அல்லது இயற்கைக் காட்சியை சேர்க்கலாம்."
    },

    "Malayalam": {

        "house":
        "വളരെ മനോഹരം! നിങ്ങൾ ഒരു വീട് വരച്ചിരിക്കുന്നു. "
        "ആദ്യം വീടിന്റെ പ്രധാന ഭാഗം ചതുരമോ ദീർഘചതുരമോ ആയി വരയ്ക്കുക. "
        "തുടർന്ന് മേൽക്കൂര ത്രികോണമായി വരയ്ക്കുക. "
        "പിന്നീട് വാതിലും ജനലുകളും ചേർക്കുക. "
        "അവസാനം മരങ്ങളും പൂക്കളും ഒരു പൂന്തോട്ടവും ചേർക്കാം.",

        "tree":
        "വളരെ മനോഹരം! നിങ്ങൾ ഒരു മരം വരച്ചിരിക്കുന്നു. "
        "ആദ്യം മരത്തിന്റെ തണ്ട് വരയ്ക്കുക. "
        "തുടർന്ന് ശാഖകൾ വരയ്ക്കുക. "
        "പിന്നീട് ഇലകൾ വരയ്ക്കുക. "
        "അവസാനം പുല്ലും പൂക്കളും പക്ഷികളും ചേർത്ത് ചിത്രം മനോഹരമാക്കുക.",

        "sun":
        "അടിപൊളി! നിങ്ങൾ സൂര്യനെ വരച്ചിരിക്കുന്നു. "
        "ആദ്യം ഒരു വൃത്തം വരയ്ക്കുക. "
        "തുടർന്ന് ചുറ്റും കിരണങ്ങൾ വരയ്ക്കുക. "
        "കിരണങ്ങൾ വ്യത്യസ്ത നീളത്തിൽ വരയ്ക്കുക. "
        "അവസാനം മേഘങ്ങളും മലകളും പ്രകൃതി ദൃശ്യങ്ങളും ചേർക്കാം."
    },

    "Hindi": {

        "house":
        "बहुत अच्छा! आपने एक घर बनाया है। "
        "सबसे पहले घर का मुख्य भाग एक वर्ग या आयत के रूप में बनाएं। "
        "फिर ऊपर त्रिकोण के आकार की छत बनाएं। "
        "इसके बाद दरवाजा और खिड़कियां बनाएं। "
        "अंत में पेड़, फूल या बगीचा बनाएं।",

        "tree":
        "बहुत अच्छा! आपने एक पेड़ बनाया है। "
        "सबसे पहले पेड़ का तना बनाएं। "
        "फिर शाखाएं बनाएं। "
        "इसके बाद पत्तियां बनाएं। "
        "अंत में घास, फूल या पक्षी बनाएं।",

        "sun":
        "बहुत बढ़िया! आपने सूरज बनाया है। "
        "सबसे पहले एक गोला बनाएं। "
        "फिर उसके चारों ओर किरणें बनाएं। "
        "किरणों को अलग-अलग लंबाई में बनाएं। "
        "अंत में बादल, पहाड़ या प्राकृतिक दृश्य बनाएं।"
    }
}


# ==============================
# DRAWING CANVAS
# ==============================

st.write("### ✏️ Draw here")

canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 0)",
    stroke_width=5,
    stroke_color="#000000",
    background_color="#FFFFFF",
    height=450,
    width=600,
    drawing_mode="freedraw",
    display_toolbar=True,

    # IMPORTANT FOR NEW VERSION
    return_image_data=True,

    key="drawmate_canvas"
)


# ==============================
# CHECK DRAWING
# ==============================

if st.button("🔍 Check My Drawing"):

    if canvas_result.image_data is None:

        st.warning(
            "✏️ Please draw something first!"
        )

    else:

        # Get drawing
        image_array = canvas_result.image_data

        # Convert to PIL
        image = Image.fromarray(
            image_array.astype("uint8")
        ).convert("L")

        # Improve contrast
        image = ImageOps.autocontrast(image)

        # Resize
        image = image.resize(
            (64, 64),
            Image.Resampling.LANCZOS
        )

        # Convert to numpy
        arr = np.array(image)

        # Invert
        arr = 255 - arr

        # Normalize
        arr = arr.astype("float32") / 255.0

        # CNN input
        arr = arr.reshape(
            1,
            64,
            64,
            1
        )

        # ==============================
        # AI PREDICTION
        # ==============================

        prediction = model.predict(
            arr,
            verbose=0
        )[0]

        predicted_index = int(
            np.argmax(prediction)
        )

        predicted_class = categories[
            predicted_index
        ]

        confidence = float(
            prediction[predicted_index] * 100
        )


        # ==============================
        # RESULT
        # ==============================

        st.divider()

        if predicted_class in [
            "house",
            "tree",
            "sun"
        ]:

            st.success(
                f"🎨 I think you drew a "
                f"**{predicted_class.upper()}**!"
            )

            st.write(
                f"🤖 AI confidence: "
                f"**{confidence:.2f}%**"
            )

            guidance = feedback[
                language
            ][predicted_class]

            st.write("### 💡 DrawMate says:")

            st.info(guidance)


            # ==============================
            # VOICE
            # ==============================

            try:

                language_codes = {
                    "English": "en",
                    "Tamil": "ta",
                    "Malayalam": "ml",
                    "Hindi": "hi"
                }

                lang_code = language_codes[
                    language
                ]

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".mp3"
                ) as audio_file:

                    tts = gTTS(
                        text=guidance,
                        lang=lang_code
                    )

                    tts.save(
                        audio_file.name
                    )

                    audio_path = audio_file.name

                st.write(
                    "### 🔊 Listen to DrawMate"
                )

                st.audio(
                    audio_path,
                    format="audio/mp3"
                )

            except Exception:

                st.warning(
                    "Voice guidance is temporarily unavailable."
                )


        else:

            st.warning(
                "🤔 I can currently understand only "
                "House, Tree, or Sun. "
                "Please try drawing one of these! "
                "🏠 🌳 ☀️"
            )
