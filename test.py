import streamlit as st
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import tempfile
import librosa
import joblib

# ==========================
# Load model
# ==========================
model = joblib.load("svm_model.pkl")  # model kamu
scaler = joblib.load("scaler.pkl")     # scaler fitur jika pakai

# ==========================
# Fungsi bantu
# ==========================
def record_audio(duration=3, fs=44100):
    st.info(f"Merekam suara selama {duration} detik...")
    audio = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()
    return audio.flatten(), fs

def extract_features(y, sr):
    # Ekstraksi MFCC sebagai contoha
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    return mfccs_mean.reshape(1, -1)

# ==========================
# Streamlit UI
# ==========================
st.title("🎤 Klasifikasi Suara")
duration = 1

if st.button("Mulai Rekam"):
    audio, sr = record_audio(duration)
    
    # Simpan sementara
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
        wav.write(tmpfile.name, sr, (audio * 32767).astype(np.int16))
        st.audio(tmpfile.name, format="audio/wav")
        
        # Ekstrak fitur & prediksi
        features = extract_features(audio, sr)
        features_scaled = scaler.transform(features)  # jika pakai scaler
        prediction = model.predict(features_scaled)[0]
        
        st.success(f"Hasil prediksi: {prediction}")
