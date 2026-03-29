# AI Analisis Gangguan Transmisi — UIT JBT

Sistem klasifikasi otomatis penyebab gangguan saluran transmisi berbasis analisis rekaman DFR (COMTRADE IEEE C37.111).

---

## Status Saat Ini

| Item | Status |
|---|---|
| File COMTRADE diproses | 492 file dari 9 UPT |
| Berhasil diklasifikasikan | 382 file (78%) |
| Tidak dapat diklasifikasikan | 110 file (parse gagal / no fault / proteksi lain) |
| Akurasi pada file berlabel | 80% (105/132 file) |
| Proteksi yang didukung | Rele Jarak (Distance 21) |
| Proteksi yang belum didukung | Diferensial (87L), Arah EF (67N) |

---

## Klasifikasi Output

| Label | Keterangan |
|---|---|
| **GANGGUAN TRANSIEN** | Gangguan singkat, AR berhasil. Penyebab: PETIR / Layang-Layang / Hewan / Benda Asing. Dilengkapi estimasi % per penyebab. |
| **GANGGUAN PERMANEN** | AR gagal, gangguan persisten. Perlu investigasi lapangan. |
| **KONDUKTOR / KERUSAKAN PERALATAN** | Perubahan impedansi antar-fasa saat AR — indikasi tower roboh / konduktor putus / CT meledak. |
| **NON-PETIR — PERLU INVESTIGASI** | Pola tidak cocok kriteria di atas. Butuh data latih lebih banyak. |

> **Catatan GANGGUAN TRANSIEN:** PETIR, Layang-Layang, Hewan, dan Benda Asing menghasilkan karakteristik gelombang yang serupa sehingga tidak dapat dibedakan dari rekaman DFR saja. Sistem menampilkan estimasi probabilitas per penyebab berdasarkan heuristik (durasi, arus puncak, rasio i0/i1, jumlah sub-fault). Konfirmasi via data cuaca atau inspeksi lapangan tetap diperlukan.

---

## Arsitektur Pipeline

```
File COMTRADE (.cfg + .dat)
        │
        ▼
┌─────────────────────────┐
│   COMTRADE Parser        │  IEEE C37.111 (1997 & 1999)
│   Channel Normalizer     │  ABB / Siemens / GE / SEL / Schneider
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   Protection Router      │  Deteksi tipe rele dari sinyal digital
│                          │  Distance zones, AR, trip phase
└────────────┬────────────┘
             │
             ▼
┌─────────────────────────┐
│   Fault Detector         │  Inception, durasi, jumlah sub-fault
│   Feature Extractor      │  Z, i0/i1, di/dt, voltage sag, peak I
└────────────┬────────────┘
             │
        ┌────┴────┐
        ▼         ▼
   Tier 1         Tier 2
   Aturan         ML Classifier
   Deterministik  (XGBoost binary:
   (rules.py)      transien vs permanen)
        │         │
        └────┬────┘
             │
             ▼
   Hasil + Confidence + Evidence
   + Estimasi % penyebab (heuristik)
```

### Tier 1 — Aturan Deterministik
Berdasarkan sinyal proteksi yang terbaca dari file COMTRADE:
- Zona trip yang operated (Z1/Z2/Z3)
- Hasil AR (berhasil / gagal)
- Perubahan impedansi antar fasa

### Tier 2 — ML Classifier
XGBoost binary classifier, dilatih dari 132 file berlabel:
- **111 PETIR** (transien)
- **21 non-PETIR** (permanen / investigasi)
- Features: fault_duration_ms, fault_count, i0/i1 ratio, voltage_sag, peak_current, reclose_status

### Estimasi Penyebab (Heuristik)
Untuk `GANGGUAN TRANSIEN`, sistem menampilkan estimasi probabilitas per penyebab menggunakan aturan fisik:
- Durasi pendek + arus tinggi → bobot PETIR naik
- Fault count ≥ 3 → bobot Layang-Layang naik
- I0/I1 tinggi + durasi pendek → bobot Hewan naik
- Durasi panjang / AR gagal → bobot Pohon naik

**Ini bukan output model ML** — hanya heuristik berbasis domain knowledge. Retraining multi-class akan dilakukan setelah data berlabel per kelas cukup (target: ≥30 sampel per kelas).

---

## Relay yang Didukung

| Merk | Model | Deteksi |
|---|---|---|
| Siemens | 7SA, 7SL (SIPROTEC 4) | Nama sinyal + rec_dev_id |
| Siemens | SIPROTEC 5 (DIGSI 5) | BM-prefix + MPI3/MPV3 channels |
| ABB | REL670, RED670 | rec_dev_id + LINE UL1/UL2/UL3 |
| GE | P442, P444, D60 | rec_dev_id + START Z/TRIP ZONE |
| Schneider | P442 | rec_dev_id |
| NR Electric | PCS900 | rec_dev_id |
| Qualitrol | DFR Eksternal | F21_REC channels |

---

## Cara Menjalankan

### Web App (Lokal)
```bash
pip install -r requirements.txt
python webapp/app.py
# Buka http://localhost:5000
```

### Klasifikasi Batch
```bash
python batch_predict.py
# Output: data/predictions/all_predictions.csv
#         data/predictions/prediction_errors.csv
```

### Melatih Ulang Model
```bash
# Setelah ada data berlabel baru di data/labels/
python models/train.py
```

---

## Deploy (Railway)

```
1. Push ke GitHub: git push origin master
2. railway.app → New Project → Deploy from GitHub
3. Pilih repo ini → Railway baca Procfile otomatis
4. Generate Domain → selesai
```

---

## Struktur Folder

```
pipeline/
├── core/
│   ├── comtrade_parser.py      # Parse COMTRADE, normalisasi channel
│   ├── channel_normalizer.py   # Mapping nama channel → VA/VB/VC/IA/IB/IC
│   ├── protection_router.py    # Deteksi tipe & zona proteksi
│   ├── fault_detector.py       # Deteksi inception & durasi gangguan
│   └── feature_extractor.py   # Ekstraksi fitur untuk ML
├── models/
│   ├── rules.py                # Aturan deterministik Tier 1
│   ├── train.py                # Training XGBoost Tier 2
│   ├── predict.py              # Inferensi end-to-end
│   ├── stage3_petir_classifier.pkl  # Model terlatih (132 sampel)
│   └── stage3_feature_columns.pkl  # Kolom fitur
├── config/
│   └── channel_mappings.json  # Mapping channel per merk relay
├── data/
│   ├── predictions/            # Output batch (gitignored)
│   └── features/               # Fitur terlatih (gitignored)
├── webapp/
│   ├── app.py                  # Flask web app
│   └── templates/
├── batch_predict.py            # Klasifikasi batch seluruh raw_data/
├── requirements.txt
├── Procfile
└── railway.json
```

---

## Roadmap

| Tahap | Status | Keterangan |
|---|---|---|
| Parser COMTRADE multi-merk | Selesai | ABB/Siemens/GE/SEL/Schneider/NR |
| Deteksi proteksi jarak | Selesai | Zone 1-3, AR, fasa trip |
| Klasifikasi biner (transien vs permanen) | Selesai | XGBoost, 80% akurasi |
| Estimasi probabilitas per penyebab | Selesai (heuristik) | Perlu retraining multi-class |
| Web app upload & klasifikasi | Selesai | Deployed ke Railway |
| Retraining multi-class | **Menunggu** | Butuh ≥30 sampel berlabel per kelas |
| Crosscheck lapangan | **Menunggu** | Stakeholder isi kolom `correct`/`notes` di all_predictions.csv |
