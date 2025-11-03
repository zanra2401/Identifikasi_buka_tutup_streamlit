import streamlit as st
import numpy as np
import librosa
import joblib
from scipy.io import wavfile

# ==========================
# Load model
# ==========================
model = joblib.load("svm_model.pkl")  # model kamu
scaler = joblib.load("scaler.pkl")     # scaler fitur jika pakai

# ==========================
# Fungsi bantu
# ==========================
def extract_features(y, sr):
    # Ekstraksi MFCC
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    return mfccs_mean.reshape(1, -1)

# ==========================
# Streamlit UI
# ==========================
st.title("Klasifikasi Suara")
st.text("DI sini saya menggunakan upload file, dikarenakan saat di deploy tidak bisa menggunakan mic")

uploaded_file = st.file_uploader("Upload file audio (WAV)", type=["wav"])

if uploaded_file is not None:
    # Baca audio
    audio, sr = librosa.load(uploaded_file, sr=16000)
    
    # Jika stereo, ubah jadi mono
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    
    st.audio(uploaded_file, format="audio/wav")
    
    # Ekstrak fitur & prediksi
    features = extract_features(audio.astype(float), sr)
    features_scaled = scaler.transform(features)  # jika pakai scaler
    prediction = model.predict(features_scaled)[0]
    
    st.success(f"Hasil prediksi: {prediction}")
