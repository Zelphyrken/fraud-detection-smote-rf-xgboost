# Implementasi Random Forest dan XGBoost berbasis SMOTE untuk Deteksi Fraud

Proyek ini mengikuti tahapan pada laporan: pengumpulan dataset, audit dan EDA, preprocessing, feature engineering, split stratified, perbandingan Random Forest/XGBoost tanpa dan dengan SMOTE, Isolation Forest sebagai pembanding, hyperparameter tuning, threshold selection, evaluasi, dan dashboard Streamlit.

## 1. Dataset

Unduh **Credit Card Fraud Detection** dari ULB melalui Kaggle:

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Ekstrak arsip dan letakkan file bernama `creditcard.csv` di:

```text
data/raw/creditcard.csv
```

Dataset yang digunakan memiliki kolom `Time`, `V1` sampai `V28`, `Amount`, dan `Class`. `Class=0` berarti normal; `Class=1` berarti fraud.

## 2. Siapkan Python di Windows / VS Code

Buka folder proyek ini di VS Code, lalu pilih **Terminal > New Terminal**. Pastikan terminal berada di folder proyek (folder yang berisi `run_experiment.py`).

Periksa Python:

```powershell
py --version
```

Buat lingkungan virtual:

```powershell
py -m venv .venv
```

Aktifkan di PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Jika PowerShell menolak aktivasi karena kebijakan skrip, alternatif yang tidak memerlukan aktivasi adalah menjalankan perintah instalasi dan kode melalui interpreter langsung:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run_experiment.py
```

Jika aktivasi berhasil, instal dependensi:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 3. Jalankan eksperimen lengkap

Dari terminal di folder proyek:

```powershell
python run_experiment.py
```

Script akan menjalankan alur berikut secara berurutan:

1. Membaca dan memvalidasi dataset.
2. Mencetak missing value, duplikat, tipe data, dan distribusi kelas.
3. Membuat grafik EDA.
4. Menghapus baris duplikat persis (opsi `DROP_EXACT_DUPLICATES=True` di awal script) dan tidak otomatis menghapus outlier.
5. Membuat fitur `Amount_log1p = log(1 + Amount)`.
6. Membagi data secara stratified menjadi train, validation, dan test (60%/20%/20% secara keseluruhan).
7. Menjalankan hyperparameter tuning untuk Random Forest tanpa SMOTE, Random Forest dengan SMOTE, XGBoost tanpa SMOTE, dan XGBoost dengan SMOTE.
8. Menerapkan SMOTE di dalam imblearn Pipeline agar hanya terjadi pada fold training saat cross-validation dan pada data training final, bukan pada validation/test.
9. Memilih threshold yang memaksimalkan F1-Score pada validation set.
10. Melatih Isolation Forest tanpa memberikan label `Class` ke metode `fit`.
11. Menghitung Precision, Recall, F1-Score, PR-AUC (Average Precision), ROC-AUC, dan confusion matrix pada test set.
12. Memilih model dashboard berdasarkan validation PR-AUC, bukan berdasarkan hasil test.

Untuk menghemat waktu, tuning default menggunakan subset stratified hingga 40.000 baris dari **data training saja**, 3 kandidat acak per eksperimen, dan 3-fold cross-validation. Untuk eksperimen final yang lebih luas, kamu dapat meningkatkan `TUNING_MAX_ROWS` dan `TUNING_ITERATIONS` di bagian konfigurasi `run_experiment.py`, lalu menjalankan ulang. Jangan mengubah konfigurasi di tengah eksperimen tanpa mencatatnya.

### Tentang duplikat

Secara default, baris yang identik dihapus sebelum split untuk mengurangi risiko baris duplikat muncul di data train dan test. Akibatnya jumlah baris dan fraud dapat menjadi lebih kecil dari angka pada pendahuluan laporan (284.807 baris dan 492 fraud). Catat angka aktual dari output run dan jelaskan keputusan preprocessing di laporan. Jika ingin membandingkan hasil dengan data mentah apa adanya, ubah `DROP_EXACT_DUPLICATES = False`, tetapi perhatikan risiko kebocoran duplikat antar-split.

## 4. Berkas hasil

Setelah sukses:

```text
results/metrics_test.csv             # tabel metrik akhir untuk laporan
results/metrics_validation.csv       # metrik validation dan threshold
results/best_model_metadata.json     # model dashboard, threshold, fitur
models/best_model.joblib             # model supervised untuk dashboard
models/supervised_models.joblib      # empat model RF/XGBoost
models/isolation_forest.joblib       # model pembanding
models/isolation_forest_scaler.joblib
reports/figures/class_distribution.png
reports/figures/amount_distribution.png
reports/figures/feature_correlations.png
reports/figures/confusion_*.png
```

Jangan isi tabel hasil laporan dengan angka dari sumber lain. Salin angka aktual dari `results/metrics_test.csv`. `PR-AUC (Average Precision)` memakai `sklearn.metrics.average_precision_score`; sebutkan definisi ini secara konsisten dalam laporan.

## 5. Jalankan dashboard Streamlit

Dashboard baru dapat digunakan setelah `run_experiment.py` selesai dan menghasilkan artefak model.

```powershell
streamlit run app.py
```

Streamlit biasanya membuka browser lokal. Dashboard mendukung:

- formulir prediksi satu transaksi dari nilai `Time`, `V1`–`V28`, dan `Amount`;
- unggah CSV untuk memprediksi banyak transaksi;
- keluaran `Safe` atau `Flagged as Fraudulent`, skor model, dan unduhan CSV hasil.

Catatan: nilai nol pada formulir hanya nilai awal demonstrasi. Fitur V1–V28 merupakan fitur hasil transformasi pada dataset, jadi masukan harus sesuai skema data. Prediksi model adalah indikasi, bukan bukti fraud.

## 6. Cara membahas hasil untuk laporan

Gunakan `metrics_test.csv` untuk tabel hasil sesuai bagian 4 laporan, serta gambar `confusion_*.png` untuk confusion matrix. Bahas sekurang-kurangnya:

- apakah SMOTE meningkatkan Recall fraud dan apa dampaknya pada Precision;
- perubahan F1-Score dan PR-AUC setelah SMOTE;
- jumlah False Positive (transaksi normal salah ditandai) dan False Negative (fraud yang terlewat);
- perbedaan Random Forest dan XGBoost;
- perbandingan dengan Isolation Forest;
- threshold yang dipilih dari validation set dan alasan penggunaan validation, bukan test, untuk memilih threshold.

Jangan mengklaim salah satu model terbaik sebelum nilai aktual hasil run tersedia.
