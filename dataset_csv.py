import pandas as pd

# Baca dataset asli
df = pd.read_csv("data/raw/creditcard.csv")

# Ambil 100 transaksi pertama
data_uji = df.head(100).copy()

# Hapus label asli agar tidak menjadi input prediksi
if "Class" in data_uji.columns:
    data_uji = data_uji.drop(columns=["Class"])

# Simpan sebagai CSV untuk dashboard
data_uji.to_csv("transaksi_uji_100.csv", index=False)

print("Jumlah transaksi:", len(data_uji))
print("File berhasil dibuat: transaksi_uji_100.csv")