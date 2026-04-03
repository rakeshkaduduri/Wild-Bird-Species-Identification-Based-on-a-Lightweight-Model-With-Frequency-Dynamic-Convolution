import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tempfile

from predict import predict_audio


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

uploaded_file = st.file_uploader(
    "Upload Bird Audio File",
    type=["ogg", "wav", "mp3"]
)

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