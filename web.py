import streamlit as st
import pickle
import numpy as np

# Load model
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("knn_model.pkl", "rb") as f:
    model = pickle.load(f)


# UI
st.title("Prediksi Performa Siswa")
st.write("Prediksi performa pelajar di sekolah berdasarkan data kehadiran, nilai ujian, dan jam belajar mandiri.")


attendance = st.number_input(
    "Persentase Kehadiran",
    min_value=0.0,
    max_value=100.0,
    value=80.0
)

score = st.number_input(
    "Total Nilai Ujian",
    min_value=0.0,
    max_value=100.0,
    value=80.0
)

study_hours = st.number_input(
    "Jam Belajar Mandiri Per minggu",
    min_value=0.0,
    value=10.0
)


if st.button("Prediksi Siswa"):

    # Urutan fitur HARUS sama seperti saat training
    input_data = np.array([
        [attendance, score, study_hours]
    ])

    # Scaling
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_scaled)

    st.success(f"Performa Siswa Terprediksi: {prediction[0]}")