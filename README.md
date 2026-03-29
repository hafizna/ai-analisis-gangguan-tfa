# AI Analisis Gangguan — TFA (UIT JBT)

Sistem klasifikasi penyebab gangguan transmisi berbasis AI menggunakan file rekaman COMTRADE dari relai jarak (distance relay) IED.

## Deskripsi

Sistem ini menganalisis file COMTRADE (.cfg + .dat) dari relai jarak SUTT dan mengklasifikasikan penyebab gangguan ke dalam kategori:

- **PETIR (TRANSIEN)** — gangguan petir, AR berhasil, tidak perlu crew lapangan
- **KONDUKTOR / KERUSAKAN PERALATAN** — perubahan fasa saat AR, diduga tower roboh / konduktor putus, CT Meledak dan sebagainya
- **GANGGUAN PERMANEN** — AR gagal, penyebab belum spesifik, perlu investigasi lapangan
- **NON-PETIR — PERLU INVESTIGASI** — bukan petir, butuh data lebih lanjut

## Arsitektur

```
File COMTRADE
    ↓
Parser COMTRADE (IEEE C37.111)
    ↓
Deteksi Proteksi (zona, trip, AR)
    ↓
Deteksi Gangguan (inception, durasi)
    ↓
Ekstraksi Fitur (Z, i0/i1, di/dt, sag)
    ↓
Tier 1 — Aturan Deterministik
    ↓ (jika tidak ada aturan)
Tier 2 — Decision Tree ML (PETIR vs Non-PETIR)
    ↓
Hasil + Confidence + Evidence
```

**Proteksi yang didukung:** Rele Jarak (Distance / 21) — SIEMENS 7SA, GE P44x, ABB REL, PCS900, NR

**Belum didukung:** Rele Diferensial (87L), Rele Arah EF (67N)

## Cara Menjalankan (Lokal)

```bash
pip install -r requirements.txt
python webapp/app.py
```

Buka http://localhost:5000

## Deploy ke Railway

1. Push folder `pipeline/` ke GitHub sebagai repository
2. Buat akun di [railway.app](https://railway.app)
3. New Project → Deploy from GitHub repo → pilih repo ini
4. Railway otomatis membaca `Procfile` dan `requirements.txt`
5. Buka URL yang diberikan Railway

## Struktur Folder

```
pipeline/
├── core/               # Parser, router proteksi, detektor gangguan, ekstraktor fitur
├── models/             # Aturan Tier 1, training Tier 2, inferensi
│   ├── rules.py        # Aturan deterministik
│   ├── train.py        # Training classifier PETIR
│   ├── predict.py      # Inferensi end-to-end
│   └── petir_tree.pkl  # Model terlatih (62 sampel, CV F1=0.85)
├── data/
│   └── features/
│       └── labeled_features.csv
├── webapp/             # Flask web app
│   ├── app.py
│   └── templates/
├── batch_extract.py    # Ekstraksi fitur batch dari raw_data/
├── requirements.txt
├── Procfile
└── railway.json
```

## Melatih Ulang Model

```bash
python batch_extract.py   # ekstrak fitur dari raw_data/ → labeled_features.csv
python models/train.py    # latih ulang → petir_tree.pkl
```

## Catatan

- Model dilatih dengan 62 sampel bersih (54 PETIR, 8 non-PETIR)
- Hanya file dari rele jarak yang dapat diklasifikasikan
- File dari rele diferensial (87L) ditampilkan notifikasi, tidak diklasifikasikan
- Konfirmasi user tersimpan di `webapp/history.csv` untuk perbaikan model ke depan
- Folder `raw_data/` tidak disertakan di repositori (terlalu besar)
