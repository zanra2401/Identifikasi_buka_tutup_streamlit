import streamlit as st
import librosa
import io
from audio_recorder_streamlit import audio_recorder
import numpy as np
import joblib
import pandas as pd

def extract_mfcc(y, sr, n_mfcc=13):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return [np.mean(mfcc, axis=1), np.std(mfcc, axis=1)]

def silent_trim(y):  
    y_trimmed, index = librosa.effects.trim(y, top_db=30)
    return y_trimmed

def extract_temo(y, sr):
    tempo = librosa.feature.tempo(y=y, sr=sr)
    return tempo

def extract_spectral_centroid(y, sr):
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    return [np.mean(centroid, axis=1), np.std(centroid, axis=1)]

def extract_rms(y, sr):
    rms = librosa.feature.rms(y=y)
    return [np.mean(rms, axis=1), np.std(rms, axis=1)]

def extract_tonetz(y, sr):
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
    return [np.mean(tonnetz, axis=1), np.std(tonnetz, axis=1)]

def extract_mel(y, sr):
    mel = librosa.feature.melspectrogram(y=y, sr=sr)
    return [np.mean(mel, axis=1), np.std(mel, axis=1)]


model = joblib.load("svm.pkl")
scaler = joblib.load("scale.pkl")


model_identifiaksi = joblib.load("svm_speaker_identifikasi.pkl")
scaler_identifiaksi = joblib.load("scale_speaker_identifikasi.pkl")

st.title("Klasifikasi Suara")
record, upload = st.tabs(["Rekam Suara", "Upload Suara"])

def process_buka_tutup(fitur):
    columns = joblib.load("columns.pkl")

    fitur_berkorelasi = joblib.load("fitur_berkorelasi.pkl")

    data_buka_tutup = pd.DataFrame([fitur], columns=columns)


    y_buka_tutup = model.predict(scaler.transform(data_buka_tutup[fitur_berkorelasi]))
    prediksi = pd.DataFrame(y_buka_tutup, columns=["Suara"])
    
    st.write(prediksi)
    st.write(pd.DataFrame(scaler.transform(data_buka_tutup[fitur_berkorelasi]), columns=[fitur_berkorelasi]))


def process_verifikasi(fitur):
    columns_identifiaksi = joblib.load("columns_speaker_identifikasi.pkl")
    fitur_berkorelasi_identifiaksi = joblib.load("fitur_berkorelasi_speaker_identifikasi.pkl")
    data_zanuar_fathan = pd.DataFrame([fitur], columns=columns_identifiaksi)
    y_zanuar_fathan = model_identifiaksi.predict_proba(scaler_identifiaksi.transform(data_zanuar_fathan[fitur_berkorelasi_identifiaksi]))

    y_zanuar_fathan = pd.DataFrame(y_zanuar_fathan,     columns=["Zanuar", "Fathan"])
    st.write(y_zanuar_fathan)
    if y_zanuar_fathan["Zanuar"][0] > 0.8:
        st.badge("Suara Terverifikasi Zanuar")
        return True
    elif y_zanuar_fathan["Fathan"][0] > 0.8:
        st.badge("Suara Terverifikasi Fathan")
        return True
    else:
        st.badge("Suara Tidak Terverifikasi", color="red")
        return False


with record:
    st.header("Rekam Suara")
    audio_bytes = audio_recorder(
        text="",
        # energy_threshold=(-1.0,1.0),
        pause_threshold=1.5 
    )
    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")
        file = io.BytesIO(audio_bytes)

        y, sr = librosa.load(file)

        y = silent_trim(y)    

        fitur = np.hstack((
            [np.mean(y)],
            [np.std(y)],
            *extract_mfcc(y, 16000),
            *extract_temo(y, 16000),
            *extract_rms(y, 16000),
            *extract_spectral_centroid(y, 16000),
            *extract_tonetz(y, 16000),
            *extract_mel(y, 16000),
        ))

        if process_verifikasi(fitur):
            process_buka_tutup(fitur)
       
with upload:
    file = st.file_uploader("Upload Audio", ["wav"])

    if file:
        st.audio(file, format="audio/wav")
        y, sr = librosa.load(file)

        y = silent_trim(y)

        fitur = np.hstack((
            [np.mean(y)],
            [np.std(y)],
            *extract_mfcc(y, 16000),
            *extract_temo(y, 16000),
            *extract_rms(y, 16000),
            *extract_spectral_centroid(y, 16000),
            *extract_tonetz(y, 16000),
            *extract_mel(y, 16000),
        ))

        if process_verifikasi(fitur):
            process_buka_tutup(fitur)

