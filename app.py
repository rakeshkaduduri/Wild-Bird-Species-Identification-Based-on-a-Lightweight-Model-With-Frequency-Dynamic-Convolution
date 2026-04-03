import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tempfile
from predict import predict_audio

import streamlit as st
import base64

# FIRST: page config
st.set_page_config(
    page_title="Bird AI",
    page_icon="🐦",
    layout="centered"
)

# SECOND: background function
def get_base64(file):
    with open(file, "rb") as f:
        return base64.b64encode(f.read()).decode()

bg = get_base64("assets/bg.webp")

st.markdown(f"""
<style>

/* -------- GLASS CARD -------- */
.glass-card {{
    background: rgba(0, 0, 0, 0.55);
    backdrop-filter: blur(15px);
    -webkit-backdrop-filter: blur(15px);
    border-radius: 20px;
    padding: 25px;
    margin-bottom: 25px;
    border: 1px solid rgba(255,255,255,0.15);
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
}}

/* -------- STRONGER RESULT BOX -------- */
.result-box {{
    background: rgba(0, 0, 0, 0.7);
    backdrop-filter: blur(20px);
    border: 1px solid rgba(0,255,150,0.3);
    padding: 20px;
    border-radius: 15px;
    text-align: center;
}}

</style>
""", unsafe_allow_html=True)

# apply CSS
st.markdown(f"""
<style>

[data-testid="stAppViewContainer"] {{
    background: 
        linear-gradient(rgba(0,0,0,0.7), rgba(0,0,0,0.7)),
        url("data:image/webp;base64,{bg}");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}}

.card {{
    background: rgba(0,0,0,0.55);
    padding: 25px;
    border-radius: 15px;
    backdrop-filter: blur(10px);
    margin-bottom: 20px;
}}

.main-title {{
    text-align: center;
    font-size: 48px;
    font-weight: 800;
    color: white;
}}

.sub-text {{
    text-align: center;
    color: #ddd;
    margin-bottom: 30px;
}}

</style>
""", unsafe_allow_html=True)


st.markdown("""
<style>

/* FULL PAGE CENTER GLASS CONTAINER */
.main-container {
    background: rgba(0, 0, 0, 0.65);
    backdrop-filter: blur(20px);
    border-radius: 20px;
    padding: 40px;
    margin-top: 30px;
    margin-bottom: 30px;
    border: 1px solid rgba(255,255,255,0.15);
}

/* OPTIONAL: inner sections spacing */
.section {
    margin-bottom: 25px;
}

</style>
""", unsafe_allow_html=True)

st.markdown(f"""
<style>

/* ---------- BACKGROUND ---------- */
[data-testid="stAppViewContainer"] {{
    background:
        linear-gradient(rgba(0,0,0,0.5), rgba(0,0,0,0.5)),
        url("data:image/webp;base64,{bg}");
    background-size: cover;
    background-position: center;
    background-attachment: fixed;
}}

/* ---------- MAIN CONTENT GLASS ---------- */
[data-testid="stMain"] > div {{
    background: rgba(255, 255, 255, 0.12);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border-radius: 20px;

    padding: 40px;
    margin: 40px auto;

    max-width: 900px;

    border: 1px solid rgba(255,255,255,0.15);
    box-shadow: 0 8px 40px rgba(0,0,0,0.5);
}}

/* ---------- TEXT IMPROVEMENT ---------- */
h1, h2, h3, h4, h5, h6 {{
    color: white;
}}

p, div {{
    color: #dddddd;
}}

</style>
""", unsafe_allow_html=True)


# -----------------------------------
# Page configuration
# -----------------------------------
st.set_page_config(
    page_title="Bird Species Identification System",
    page_icon="🐦",
    layout="centered"
)

# -----------------------------------
# Custom CSS for centered title
# -----------------------------------
st.markdown(
    """
    <style>
    .title-center {
        text-align: center;
        font-size: 42px;
        font-weight: 700;
        white-space: nowrap;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# -----------------------------------
# Title
# -----------------------------------
st.markdown(
"""
<h1 style='text-align: center; white-space: nowrap;'>
🐦 Bird Species Identification System
</h1>
""",
unsafe_allow_html=True
)

# -----------------------------------
# Description
# -----------------------------------
st.markdown(
"""
<div style='text-align:center; font-size:18px;'>
Upload a <b>bird sound recording</b>, and the system will analyze the audio
and predict the <b>bird species</b> using a trained deep learning model.
</div>
""",
unsafe_allow_html=True
)

st.divider()


# -----------------------------------
# Upload section
# -----------------------------------
st.subheader("📂 Upload Bird Audio")

st.markdown("<div class='glass-card'>", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload Bird Audio",
    type=["wav", "mp3", "ogg"]
)

st.markdown("</div>", unsafe_allow_html=True)

st.divider()


# -----------------------------------
# When file is uploaded
# -----------------------------------
if uploaded_file is not None:

    # -----------------------------------
    # Uploaded audio
    # -----------------------------------
    st.subheader("🎧 Uploaded Audio")

    st.audio(uploaded_file)

    # Save temporary file
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        audio_path = tmp_file.name

    st.divider()

    # -----------------------------------
    # Spectrogram visualization
    # -----------------------------------
    st.subheader("📊 Spectrogram Visualization")

    y, sr = librosa.load(audio_path)

    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    mel_db = librosa.power_to_db(mel)

    fig, ax = plt.subplots(figsize=(8,4))

    librosa.display.specshow(
        mel_db,
        sr=sr,
        x_axis="time",
        y_axis="mel",
        ax=ax
    )

    ax.set_title("Mel Spectrogram")

    st.pyplot(fig)

    st.divider()

    # -----------------------------------
    # Prediction section
    # -----------------------------------
    st.subheader("🤖 Model Prediction")

    with st.spinner("Analyzing bird sound..."):
        species, confidence = predict_audio(audio_path)

    confidence_percent = confidence * 100

    st.success("Prediction completed!")

    st.markdown("### 🧠 Prediction Result")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            label="Predicted Bird Species",
            value=species
        )

    with col2:
        st.metric(
            label="Confidence",
            value=f"{confidence_percent:.2f}%"
        )

    st.divider()


# -----------------------------------
# Footer
# -----------------------------------
st.caption(
    "Bird Species Identification using Deep Learning"
)