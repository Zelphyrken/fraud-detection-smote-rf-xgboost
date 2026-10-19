"""Streamlit prototype for the trained fraud-detection model.

Run from the project root after running `python run_experiment.py`:
    streamlit run app.py
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.common import (
    BASE_DIR,
    ENGINEERED_FEATURES,
    MODEL_DIR,
    RAW_FEATURES,
    RESULTS_DIR,
    engineer_features,
)

MODEL_PATH = MODEL_DIR / "best_model.joblib"
METADATA_PATH = RESULTS_DIR / "best_model_metadata.json"

st.set_page_config(page_title="Deteksi Fraud Transaksi Digital", page_icon="🛡️", layout="wide")
st.title("Deteksi Kecurangan Transaksi Keuangan Digital")
st.caption("Prototipe penelitian: Random Forest / XGBoost dengan atau tanpa SMOTE")
st.warning(
    "Hasil model adalah indikasi untuk membantu pemeriksaan, bukan bukti pasti terjadinya fraud. "
    "Jangan gunakan prototipe ini untuk memblokir transaksi riil tanpa validasi dan pengamanan tambahan."
)

if not MODEL_PATH.exists() or not METADATA_PATH.exists():
    st.error("Model belum ditemukan. Jalankan `python run_experiment.py` dari folder proyek terlebih dahulu.")
    st.stop()

@st.cache_resource
def load_model_and_metadata():
    model = joblib.load(MODEL_PATH)
    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        metadata = json.load(file)
    return model, metadata

model, metadata = load_model_and_metadata()
threshold = float(metadata["threshold"])
st.info(
    f"Model yang dipilih berdasarkan validation PR-AUC: **{metadata['selected_model']}**. "
    f"Threshold klasifikasi: **{threshold:.6f}**."
)

st.subheader("A. Prediksi satu transaksi")
st.write(
    "Masukkan nilai fitur dari dataset. Kolom V1 sampai V28 adalah fitur numerik yang telah "
    "ditransformasikan pada dataset ULB, bukan nama pelanggan, toko, atau lokasi transaksi."
)
with st.form("single_transaction_form"):
    st.write("Fitur inti")
    top_cols = st.columns(3)
    entered = {}
    for index, feature in enumerate(RAW_FEATURES):
        col = top_cols[index % 3]
        default_value = 0.0
        step = 1.0 if feature in {"Time", "Amount"} else 0.1
        entered[feature] = col.number_input(
            feature,
            value=default_value,
            step=step,
            format="%.6f",
            key=f"input_{feature}",
        )
    submitted = st.form_submit_button("Prediksi transaksi")

if submitted:
    row = pd.DataFrame([{feature: entered[feature] for feature in RAW_FEATURES}])
    try:
        model_input = engineer_features(row).loc[:, ENGINEERED_FEATURES]
        fraud_probability = float(model.predict_proba(model_input)[:, 1][0])
        flagged = fraud_probability >= threshold
        left, right = st.columns(2)
        left.metric("Estimasi skor kelas fraud", f"{fraud_probability:.2%}")
        right.metric("Threshold model", f"{threshold:.2%}")
        if flagged:
            st.error("**Flagged as Fraudulent** — transaksi ditandai untuk pemeriksaan lebih lanjut.")
        else:
            st.success("**Safe** — model tidak menandai transaksi ini sebagai fraud pada threshold terpilih.")
    except Exception as error:
        st.exception(error)

st.divider()
st.subheader("B. Prediksi banyak transaksi melalui CSV")
st.write(
    "CSV harus memiliki kolom `Time`, `V1` sampai `V28`, dan `Amount`. Jika CSV juga memiliki `Class`, "
    "kolom itu akan diabaikan untuk prediksi. Gunakan data berizin dan jangan unggah data pribadi sensitif."
)
uploaded_file = st.file_uploader("Unggah CSV transaksi", type=["csv"])
if uploaded_file is not None:
    try:
        raw_df = pd.read_csv(uploaded_file)
        # Prevent the target from accidentally being used as a model input.
        if "Class" in raw_df.columns:
            raw_df = raw_df.drop(columns=["Class"])
        missing = [column for column in RAW_FEATURES if column not in raw_df.columns]
        if missing:
            st.error("Kolom belum tersedia: " + ", ".join(missing))
        else:
            model_input = engineer_features(raw_df)
            probabilities = model.predict_proba(model_input)[:, 1]
            labels = (probabilities >= threshold).astype(int)
            output = raw_df.loc[:, RAW_FEATURES].copy()
            output["Fraud_Score"] = probabilities
            output["Prediction"] = [
                "Flagged as Fraudulent" if label == 1 else "Safe" for label in labels
            ]
            st.write("Ringkasan prediksi:")
            st.dataframe(output.head(100), use_container_width=True)
            st.caption(f"Menampilkan maksimum 100 baris pertama dari {len(output):,} transaksi.")
            st.download_button(
                "Unduh hasil prediksi CSV",
                data=output.to_csv(index=False).encode("utf-8"),
                file_name="hasil_prediksi_fraud.csv",
                mime="text/csv",
            )
    except Exception as error:
        st.exception(error)

with st.expander("Catatan metodologi"):
    st.markdown(
        "- SMOTE diterapkan hanya pada fold/training data di pipeline training.\n"
        "- Threshold dipilih menggunakan validation set dengan tujuan memaksimalkan F1-Score.\n"
        "- Testing set hanya digunakan untuk evaluasi akhir; model dashboard dipilih berdasarkan validation PR-AUC.\n"
        "- Data dari formulir default nol hanya cocok untuk demonstrasi teknis, bukan contoh transaksi dunia nyata."
    )
