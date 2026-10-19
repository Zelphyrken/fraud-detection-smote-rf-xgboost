# Deteksi Fraud dengan SMOTE, Random Forest, dan XGBoost

Proyek Python untuk membandingkan Random Forest dan XGBoost, masing-masing tanpa dan dengan SMOTE, pada dataset transaksi kartu kredit. Isolation Forest digunakan sebagai model pembanding. Hasil training mencakup metrik evaluasi, grafik, dan model yang bisa digunakan melalui dashboard Streamlit.

Urutan menjalankan proyek: siapkan Python dan folder proyek, unduh dataset, instal dependensi sesuai sistem operasi, jalankan training, lalu buka dashboard.

## Daftar isi

- [Kebutuhan](#kebutuhan)
- [Siapkan folder proyek](#siapkan-folder-proyek)
- [Siapkan dataset](#siapkan-dataset)
- [Windows](#windows)
- [Linux](#linux)
- [macOS](#macos)
- [Alur eksperimen dan konfigurasi](#alur-eksperimen-dan-konfigurasi)
- [File hasil eksperimen](#file-hasil-eksperimen)
- [Cara menggunakan dashboard](#cara-menggunakan-dashboard)
- [Menjalankan ulang dan menghentikan aplikasi](#menjalankan-ulang-dan-menghentikan-aplikasi)
- [Mengatasi masalah](#mengatasi-masalah)
- [Membaca hasil untuk laporan](#membaca-hasil-untuk-laporan)

## Kebutuhan

- Python 64-bit. Gunakan Python 3.12 untuk mengikuti contoh Windows dan macOS di bawah. Dependensi berhasil diimpor pada Python 3.12 di lingkungan Linux proyek ini.
- Git jika ingin mengambil proyek dengan `git clone`. Bisa juga mengunduh ZIP dari GitHub.
- Koneksi internet untuk mengunduh dataset dan dependensi.
- Browser untuk membuka dashboard. VS Code bersifat opsional.

Training menggunakan CPU; GPU dan CUDA tidak diperlukan. Waktu training dan penggunaan RAM bergantung pada perangkat serta konfigurasi eksperimen.

Daftar dependensi ada di [requirements.txt](requirements.txt): pandas, NumPy, scikit-learn, imbalanced-learn, XGBoost, Matplotlib, seaborn, joblib, dan Streamlit. Versinya memakai batas minimum, sehingga versi yang terpasang dapat berbeda antarwaktu instalasi.

## Siapkan folder proyek

Jika Git sudah terpasang, jalankan di PowerShell, CMD, atau terminal Linux/macOS:

```sh
git clone https://github.com/Zelphyrken/fraud-detection-smote-rf-xgboost.git
cd fraud-detection-smote-rf-xgboost
```

Jika memakai ZIP, unduh melalui **Code > Download ZIP** di [repositori GitHub](https://github.com/Zelphyrken/fraud-detection-smote-rf-xgboost), lalu ekstrak. Buka terminal di folder hasil ekstrak. Nama folder ZIP dapat memiliki akhiran nama branch.

Jika proyek sudah ada di komputer, langsung buka foldernya. Semua perintah proyek di README ini dijalankan dari folder yang berisi `app.py`, `run_experiment.py`, dan `requirements.txt`.

Struktur utama:

```text
fraud-detection-smote-rf-xgboost/
├── app.py                  # Dashboard Streamlit
├── run_experiment.py       # Training dan evaluasi
├── requirements.txt       # Dependensi Python
├── src/
│   └── common.py           # Path, skema fitur, dan fungsi bersama
├── data/
│   └── raw/
│       └── creditcard.csv  # Dataset, unduh terpisah
├── models/                 # Model hasil training
├── results/                # Metrik dan metadata model
├── reports/
│   └── figures/            # Grafik hasil eksperimen
└── hasil-tes/              # Contoh hasil prediksi CSV
```

Script membuat folder output jika belum ada. Dataset, model, dan hasil eksperimen diabaikan oleh `.gitignore`, sehingga perlu disiapkan atau dihasilkan di komputer sendiri.

## Siapkan dataset

1. Buka [Credit Card Fraud Detection di Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
2. Login jika diminta, lalu unduh dataset.
3. Ekstrak arsip unduhan.
4. Salin file `creditcard.csv` ke `data/raw/creditcard.csv` di dalam folder proyek.

Pastikan nama file tepat `creditcard.csv`, bukan `creditcard.csv.zip` atau `creditcard.csv.csv`.

Dataset training harus berisi kolom berikut:

| Kolom | Keterangan |
| --- | --- |
| `Time` | Waktu relatif dalam detik pada dataset, bukan jam transaksi. |
| `V1` sampai `V28` | Fitur numerik hasil transformasi dataset ULB. |
| `Amount` | Nominal transaksi. |
| `Class` | Label aktual: `0` normal, `1` fraud. |

Gunakan dataset lengkap untuk training. File di `hasil-tes/` merupakan hasil prediksi, sehingga tidak menggantikan dataset training berlabel `Class`.

## Windows

### 1. Instal Python dan Git

Instal Python 3.12 64-bit dari [situs Python](https://www.python.org/downloads/windows/). Jika menggunakan installer klasik, aktifkan opsi **Add python.exe to PATH** dan Python Launcher jika tersedia. Instal [Git for Windows](https://git-scm.com/downloads/win) jika memakai `git clone`, lalu buka ulang terminal.

Periksa Python:

```powershell
py -3.12 --version
```

Jika `py` tidak tersedia tetapi `python --version` menunjukkan versi yang ingin digunakan, ganti `py -3.12` dengan `python` pada perintah pembuatan virtual environment.

### 2. Buat virtual environment

Dari folder proyek:

```powershell
py -3.12 -m venv .venv
```

Virtual environment menyimpan dependensi proyek di folder `.venv`.

### 3. Aktifkan environment

Untuk PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Untuk Command Prompt (CMD):

```bat
.venv\Scripts\activate.bat
```

Jika PowerShell menampilkan pesan `running scripts is disabled`, gunakan Python environment secara langsung tanpa aktivasi:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run_experiment.py
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Jalankan setiap perintah setelah perintah sebelumnya selesai tanpa error. Cara ini sudah mencakup instalasi, training, dan dashboard; lanjut ke [cara menggunakan dashboard](#cara-menggunakan-dashboard) setelah training selesai dan dashboard berjalan.

### 4. Instal dependensi

Jika environment berhasil diaktifkan:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

### 5. Jalankan training dan dashboard

Pastikan dataset sudah tersedia, lalu jalankan:

```powershell
python run_experiment.py
```

Tunggu sampai terminal menampilkan `LANGKAH 8: Selesai: file hasil eksperimen`. Setelah itu jalankan:

```powershell
python -m streamlit run app.py
```

Buka alamat lokal yang muncul di terminal, biasanya [http://localhost:8501](http://localhost:8501).

## Linux

### 1. Instal Python dan paket sistem

Pilih perintah sesuai distro yang digunakan.

#### Ubuntu / Debian (`apt`)

```bash
sudo apt update
sudo apt install python3 python3-venv python3-pip git libgomp1
```

Paket `python3-venv` menyediakan dukungan virtual environment. Paket `libgomp1` menyediakan runtime OpenMP untuk XGBoost.

#### Fedora (`dnf`)

```bash
sudo dnf install python3 python3-pip git libgomp
```

Modul `venv` tersedia bersama Python. Paket `libgomp` menyediakan runtime OpenMP.

#### Arch Linux (`pacman`)

```bash
sudo pacman -Syu --needed python python-pip git libgomp
```

Perintah ini memperbarui sistem sekaligus menginstal paket yang diperlukan. Paket `python` sudah menyertakan modul `venv`; `python-pip` menyediakan pip. Paket [libgomp](https://archlinux.org/packages/core/x86_64/libgomp/) menyediakan runtime OpenMP.

Setelah instalasi pada distro yang dipilih, periksa versi Python:

```bash
python3 --version
```

Versi Python mengikuti repositori distro saat instalasi. Jika dependensi belum mendukung versi tersebut, gunakan Python 3.12 yang dipasang terpisah dari Python sistem, lalu buat environment dengan `python3.12 -m venv .venv`. Pada Ubuntu/Debian, siapkan juga paket `venv` untuk versi Python tersebut.

Referensi paket: [python3-venv Ubuntu](https://packages.ubuntu.com/noble/python3-venv), [libgomp Fedora](https://packages.fedoraproject.org/pkgs/gcc/libgomp/), [Python Arch](https://archlinux.org/packages/core/x86_64/python/), dan [python-pip Arch](https://archlinux.org/packages/extra/any/python-pip/).

### 2. Buat dan aktifkan virtual environment

Dari folder proyek, gunakan Bash atau Zsh. Langkah berikut berlaku untuk ketiga distro:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Setelah environment aktif, gunakan perintah `python` untuk instalasi dependensi, training, dan dashboard.

Jika menggunakan Fish, ganti perintah aktivasi dengan:

```fish
source .venv/bin/activate.fish
```

### 3. Instal dependensi

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

Jalankan instalasi dependensi di dalam `.venv`; perintah pip di atas tidak memerlukan `sudo`.

### 4. Jalankan training dan dashboard

Pastikan dataset sudah tersedia, lalu jalankan:

```bash
python run_experiment.py
```

Tunggu sampai terminal menampilkan `LANGKAH 8: Selesai: file hasil eksperimen`. Setelah itu jalankan:

```bash
python -m streamlit run app.py
```

Buka alamat lokal yang muncul di terminal, biasanya [http://localhost:8501](http://localhost:8501).

## macOS

### 1. Instal Python dan OpenMP

Panduan ini menggunakan Homebrew. Jika belum terpasang, ikuti petunjuk di [situs Homebrew](https://brew.sh/), termasuk langkah `brew shellenv` yang ditampilkan setelah instalasi. Buka ulang Terminal, lalu jalankan:

```bash
brew install python@3.12 libomp git
python3.12 --version
```

Homebrew menyediakan perintah `python3.12` melalui [paket python@3.12](https://formulae.brew.sh/formula/python@3.12). Paket `libomp` menyediakan runtime OpenMP untuk XGBoost. Pada Apple Silicon, gunakan Python dan Homebrew dengan arsitektur yang sama agar library dapat dimuat.

### 2. Buat dan aktifkan virtual environment

Dari folder proyek, gunakan Terminal dengan Zsh atau Bash:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### 3. Instal dependensi

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip check
```

### 4. Jalankan training dan dashboard

Pastikan dataset sudah tersedia, lalu jalankan:

```bash
python run_experiment.py
```

Tunggu sampai terminal menampilkan `LANGKAH 8: Selesai: file hasil eksperimen`. Setelah itu jalankan:

```bash
python -m streamlit run app.py
```

Buka alamat lokal yang muncul di terminal, biasanya [http://localhost:8501](http://localhost:8501).

## Alur eksperimen dan konfigurasi

`run_experiment.py` menjalankan tahapan berikut:

1. Membaca dataset dan memeriksa kolom wajib, missing value, duplikat, tipe data, serta distribusi kelas.
2. Menghapus baris duplikat persis sesuai konfigurasi dan menyimpan grafik EDA. Outlier tidak dihapus otomatis.
3. Menambahkan fitur `Amount_log1p = log(1 + Amount)`.
4. Membagi data secara stratified menjadi train 60%, validation 20%, dan test 20%.
5. Melakukan tuning Random Forest dan XGBoost, masing-masing tanpa dan dengan SMOTE. Scaling dan SMOTE berada dalam pipeline; keduanya hanya dipelajari atau diterapkan pada data training setiap fold.
6. Melatih ulang setiap model dengan parameter terpilih pada seluruh bagian training, lalu memilih threshold yang memaksimalkan F1-Score pada validation set.
7. Melatih Isolation Forest tanpa label pada proses `fit`, lalu memilih threshold menggunakan validation set.
8. Mengevaluasi semua model pada test set dan menyimpan hasil. Model dashboard dipilih dari empat model supervised berdasarkan validation PR-AUC.

Konfigurasi tersedia di bagian awal [run_experiment.py](run_experiment.py):

| Pengaturan | Nilai awal | Fungsi |
| --- | --- | --- |
| `DROP_EXACT_DUPLICATES` | `True` | Menghapus baris identik sebelum pembagian data. |
| `TEST_SIZE` | `0.20` | Menyisihkan 20% data untuk test. |
| `VALIDATION_SIZE` | `0.25` | Mengambil 25% dari sisa 80%, sehingga validation menjadi 20% total. |
| `TUNING_MAX_ROWS` | `40_000` | Batas baris subset stratified dari training untuk tuning. |
| `TUNING_ITERATIONS` | `3` | Jumlah kandidat parameter per eksperimen supervised. |
| `CV_FOLDS` | `3` | Jumlah fold cross-validation. |
| `SEARCH_N_JOBS` | `2` | Jumlah proses paralel untuk pencarian parameter. |
| `RANDOM_STATE` | `42` | Seed untuk proses acak. |

Untuk mengurangi beban pencarian parameter, turunkan `SEARCH_N_JOBS` menjadi `1`. Untuk pencarian yang lebih luas, naikkan `TUNING_MAX_ROWS` dan `TUNING_ITERATIONS`. Model final tetap dilatih pada seluruh bagian training; pengurangan subset tuning tidak mengurangi data pada tahap ini.

Catat perubahan konfigurasi sebelum menjalankan ulang. Penghapusan duplikat membuat jumlah baris dan fraud berbeda dari dataset mentah. Gunakan jumlah aktual yang tercetak di terminal. Jika `DROP_EXACT_DUPLICATES` diubah menjadi `False`, baris identik dapat masuk ke bagian train dan test.

## File hasil eksperimen

| File | Isi |
| --- | --- |
| `results/metrics_test.csv` | Precision, Recall, F1-Score, PR-AUC/AP, ROC-AUC, dan threshold untuk lima eksperimen. |
| `results/metrics_validation.csv` | Metrik validation, threshold, dan skor CV untuk empat eksperimen supervised. |
| `results/best_model_metadata.json` | Nama model dashboard, threshold, daftar fitur, dan informasi pemilihan model. |
| `models/best_model.joblib` | Model supervised yang digunakan dashboard. |
| `models/supervised_models.joblib` | Empat model Random Forest/XGBoost beserta threshold dan parameter. |
| `models/isolation_forest.joblib` | Model Isolation Forest. |
| `models/isolation_forest_scaler.joblib` | Scaler untuk Isolation Forest. |
| `reports/figures/class_distribution.png` | Distribusi kelas. |
| `reports/figures/amount_distribution.png` | Distribusi `log(1 + Amount)` per kelas. |
| `reports/figures/feature_correlations.png` | Korelasi fitur terhadap `Class`. |
| `reports/figures/confusion_*.png` | Confusion matrix setiap eksperimen. |

Menjalankan eksperimen lagi akan menimpa output dengan nama yang sama. Salin hasil sebelumnya ke folder lain jika ingin membandingkan beberapa konfigurasi.

## Cara menggunakan dashboard

Dashboard memerlukan `models/best_model.joblib` dan `results/best_model_metadata.json` dari training yang sama. Training cukup dilakukan sekali selama model dan konfigurasi belum berubah.

### Prediksi satu transaksi

1. Buka bagian **A. Prediksi satu transaksi**.
2. Isi `Time`, `V1` sampai `V28`, dan `Amount` dari satu transaksi pada dataset.
3. Klik **Prediksi transaksi**.
4. Baca skor fraud, threshold, dan status prediksi.

Nilai nol pada formulir adalah nilai awal demonstrasi. Fitur `V1` sampai `V28` harus mengikuti representasi dataset ULB.

### Prediksi banyak transaksi dari CSV

CSV input wajib memiliki 30 kolom fitur berikut:

```csv
Time,V1,V2,V3,V4,V5,V6,V7,V8,V9,V10,V11,V12,V13,V14,V15,V16,V17,V18,V19,V20,V21,V22,V23,V24,V25,V26,V27,V28,Amount
```

Gunakan pemisah koma, titik sebagai tanda desimal, dan nilai numerik tanpa sel kosong. Kolom `Class` bersifat opsional dan diabaikan saat prediksi. Fitur `Amount_log1p` dibuat otomatis.

Untuk membuat CSV uji berisi 100 transaksi pertama dari dataset, jalankan dari folder proyek dengan environment aktif:

```sh
python -c "import pandas as pd; from pathlib import Path; Path('results').mkdir(exist_ok=True); pd.read_csv('data/raw/creditcard.csv', nrows=100).drop(columns=['Class']).to_csv('results/input_100_transaksi.csv', index=False)"
```

1. Buka bagian **B. Prediksi banyak transaksi melalui CSV**.
2. Unggah `results/input_100_transaksi.csv` atau CSV lain dengan kolom yang sesuai.
3. Dashboard memprediksi seluruh baris dan menampilkan maksimal 100 baris pertama.
4. Klik **Unduh hasil prediksi CSV** untuk menyimpan seluruh hasil sebagai `hasil_prediksi_fraud.csv`.

Output menambahkan kolom berikut:

| Kolom | Keterangan |
| --- | --- |
| `Fraud_Score` | Skor kelas fraud dari model. |
| `Prediction` | `Flagged as Fraudulent` jika skor lebih besar atau sama dengan threshold; selain itu `Safe`. |

CSV hasil prediksi belum menjadi evaluasi performa. Perbandingan dengan label aktual `Class` tetap diperlukan untuk menghitung metrik. Hasil model merupakan indikasi untuk pemeriksaan transaksi, bukan bukti pasti fraud.

## Menjalankan ulang dan menghentikan aplikasi

Setiap membuka terminal baru, masuk ke folder proyek dan aktifkan `.venv` sesuai sistem operasi. Jika dependensi dan artefak model masih tersedia, langsung jalankan:

```sh
python -m streamlit run app.py
```

Tekan `Ctrl+C` di terminal untuk menghentikan dashboard atau training. Training yang dihentikan perlu dijalankan ulang dari awal.

Setelah selesai menggunakan environment, jalankan:

```sh
deactivate
```

Di VS Code, buka folder proyek, pilih **Python: Select Interpreter** dari Command Palette, lalu pilih Python di `.venv`. Gunakan **Terminal > New Terminal** untuk menjalankan perintah. Pada Windows, interpreter berada di `.venv\Scripts\python.exe`; pada Linux/macOS di `.venv/bin/python`.

## Mengatasi masalah

| Masalah | Langkah perbaikan |
| --- | --- |
| `py`, `python`, atau `python3` tidak ditemukan | Pastikan Python sudah terpasang, buka ulang terminal, lalu gunakan nama perintah sesuai bagian sistem operasi. Di Windows, periksa PATH atau gunakan Python Launcher. |
| `No module named venv` atau `ensurepip is not available` di Ubuntu/Debian | Jalankan `sudo apt install python3-venv`, lalu buat ulang environment. Jika memakai versi Python khusus, instal paket `venv` untuk versi tersebut. |
| Environment gagal digunakan setelah pembaruan Python di Arch | Buat ulang `.venv` dengan versi Python yang dipilih, lalu instal ulang `requirements.txt`. Folder environment lama dapat merujuk ke versi Python sebelum pembaruan. |
| `externally-managed-environment` | Buat dan aktifkan `.venv`, lalu ulangi instalasi dependensi melalui `python -m pip`. |
| `ModuleNotFoundError` atau Streamlit tidak ditemukan | Aktifkan `.venv`, jalankan `python -m pip install -r requirements.txt`, lalu gunakan `python -m streamlit run app.py`. |
| PowerShell menolak `Activate.ps1` | Gunakan `.\.venv\Scripts\python.exe` secara langsung seperti pada panduan Windows. |
| `Dataset belum ditemukan` | Pastikan file sudah diekstrak dan berada di `data/raw/creditcard.csv`. |
| `Dataset tidak memiliki kolom wajib` atau missing value | Periksa header dan isi dataset. Gunakan dataset Kaggle yang sesuai dan lengkap untuk training. |
| Dashboard menampilkan `Model belum ditemukan` | Jalankan `python run_experiment.py` sampai selesai dan periksa kedua file model serta metadata dashboard. |
| XGBoost gagal dimuat karena DLL di Windows | Instal [Microsoft Visual C++ Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist) sesuai arsitektur Python, lalu buka ulang terminal. Runtime ini diperlukan menurut [panduan instalasi XGBoost](https://xgboost.readthedocs.io/en/stable/install.html). |
| Error `libomp.dylib` di macOS | Jalankan `brew install libomp`. Periksa juga kesesuaian arsitektur Python dan Homebrew. |
| Error `libgomp.so.1` di Linux | Ubuntu/Debian: `sudo apt install libgomp1`. Fedora: `sudo dnf install libgomp`. Arch: `sudo pacman -Syu --needed libgomp`. |
| `No matching distribution found` atau pip mencoba build lalu gagal | Periksa versi dan arsitektur Python. Coba Python 3.12 64-bit dengan `.venv` baru, upgrade pip, lalu instal dependensi lagi. |
| Training kehabisan RAM atau terlalu berat | Ubah `SEARCH_N_JOBS = 1`. Mengurangi subset/kandidat tuning membantu tahap pencarian, tetapi training final tetap menggunakan seluruh bagian training. |
| Port `8501` sudah digunakan | Jalankan `python -m streamlit run app.py --server.port 8502`, lalu buka alamat yang tercetak. |
| Browser tidak terbuka otomatis | Buka URL lokal yang ditampilkan terminal dan biarkan proses Streamlit tetap berjalan. |
| Model gagal dimuat setelah dependensi berubah | Gunakan versi dependensi saat training atau jalankan ulang eksperimen dengan environment saat ini. Setelah training ulang, hentikan dan mulai ulang dashboard agar cache model dimuat ulang. |

Untuk memeriksa interpreter yang sedang digunakan:

```sh
python -c "import sys; print(sys.executable)"
```

Path seharusnya mengarah ke `.venv` proyek. Referensi perintah environment tersedia di [dokumentasi Python venv](https://docs.python.org/3.12/library/venv.html), dan perintah menjalankan dashboard tersedia di [dokumentasi Streamlit](https://docs.streamlit.io/develop/concepts/architecture/run-your-app).

## Membaca hasil untuk laporan

Gunakan angka aktual dari `results/metrics_test.csv` dan gambar `reports/figures/confusion_*.png`. Bahas pengaruh SMOTE terhadap Precision, Recall, F1-Score, dan PR-AUC, lalu bandingkan Random Forest, XGBoost, serta Isolation Forest.

`PR-AUC (Average Precision)` pada proyek ini dihitung dengan `sklearn.metrics.average_precision_score`. Gunakan definisi tersebut secara konsisten. Confusion matrix memakai susunan `[[TN, FP], [FN, TP]]`; perhatikan transaksi normal yang salah ditandai (`FP`) dan fraud yang terlewat (`FN`).

Threshold dipilih dari validation set untuk memaksimalkan F1-Score. Test set digunakan untuk evaluasi akhir, sedangkan pemilihan model dashboard memakai validation PR-AUC. Catat jumlah data setelah preprocessing dan konfigurasi eksperimen agar hasil laporan dapat ditelusuri.
