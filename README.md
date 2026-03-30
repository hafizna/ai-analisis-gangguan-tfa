# AI Analisis Gangguan Transmisi

Sistem klasifikasi otomatis penyebab gangguan saluran transmisi berbasis analisis rekaman DFR (COMTRADE IEEE C37.111).

---

## Pencapaian Utama

Tantangan terbesar dalam membangun sistem ini bukan pada algoritmanya, melainkan pada **heterogenitas data mentah**: setiap merk relay menyimpan nama channel, format sinyal digital, dan struktur file COMTRADE secara berbeda. Tanpa normalisasi yang benar, fitur apapun tidak bisa diekstrak.

### Yang sudah berhasil diselesaikan:

**1. Parser & Normalisasi Multi-Merk (fondasi sistem)**
Sistem berhasil membaca dan menormalisasi rekaman COMTRADE dari 6+ merk relay yang berbeda secara otomatis — tanpa konfigurasi manual per file:

| Merk | Konvensi channel yang ditangani |
|---|---|
| Siemens SIPROTEC 4 | Nama sinyal generik (IA, UL1, dsb.) |
| Siemens SIPROTEC 5 (DIGSI 5) | Format komponen `MPI3p1:I A`, `MPV3p1:V A` |
| ABB REL670 / RED670 | `LINE UL1/UL2/UL3`, `LINE UN` |
| GE / Multilin | `START Z1`, `TRIP ZONE`, format IL1–IL3 |
| Schneider / NR Electric | Format lokal PLN (`CTR`, `VTR`, dsb.) |
| Qualitrol DFR Eksternal | Channel `F21_REC` + format RST |

Normalisasi ini menyelesaikan masalah yang selama ini membuat analisis DFR bergantung pada penanganan manual per rekaman.

**2. Deteksi Proteksi & Ekstraksi Fitur Otomatis**
Dari rekaman mentah, sistem secara otomatis mengidentifikasi:
- Tipe proteksi yang bekerja (rele jarak Z1/Z2/Z3, AR, trip fasa)
- Inception time, durasi, dan jumlah sub-fault
- Fitur kuantitatif: di/dt, arus puncak, rasio i0/i1, voltage sag, impedansi

**3. Klasifikasi dengan Akurasi 80% pada Data Berlabel**
Dari 492 file COMTRADE yang diproses dari 9 UPT:

| Item | Hasil |
|---|---|
| Berhasil diklasifikasikan | 382 file (78%) |
| Tidak dapat diklasifikasikan | 110 file (parse gagal / no fault / proteksi lain) |
| Akurasi pada 132 file berlabel | **80%** (105/132 file tepat) |
| Proteksi yang didukung | Rele Jarak (Distance 21) |
| Proteksi yang belum didukung | Diferensial (87L), Arah EF (67N) |

Angka 78% berhasil diklasifikasikan dari raw data yang belum pernah disentuh sebelumnya — dengan variasi merk relay, kualitas rekaman, dan kondisi gangguan yang beragam — merupakan validasi bahwa fondasi pipeline sudah solid.

---

## Keterbatasan Saat Ini & Arah Pengembangan

Klasifikasi saat ini membedakan **gangguan transien vs. permanen** (binary). Pemisahan lebih lanjut per penyebab transien (PETIR / Layang-Layang / Hewan / Benda Asing) belum dilakukan via ML karena keterbatasan data berlabel, **bukan karena keterbatasan teknis**.

Setiap penyebab transien memiliki karakteristik gelombang yang dapat dibedakan — pola ini sudah teramati dari data yang ada dan sudah diimplementasikan sebagai heuristik. Yang dibutuhkan untuk meningkatkannya ke classifier statistik adalah jumlah sampel berlabel yang cukup per kelas (lihat distribusi data di bawah).

---

## Klasifikasi Output

---

## Klasifikasi Output

| Label | Keterangan |
|---|---|
| **GANGGUAN TRANSIEN** | Gangguan singkat, AR berhasil. Penyebab: PETIR / Layang-Layang / Hewan / Benda Asing. Dilengkapi estimasi % per penyebab. |
| **GANGGUAN PERMANEN** | AR gagal, gangguan persisten. Perlu investigasi lapangan. |
| **KONDUKTOR / KERUSAKAN PERALATAN** | Perubahan impedansi antar-fasa saat AR — indikasi tower roboh / konduktor putus / CT meledak. |
| **NON-PETIR — PERLU INVESTIGASI** | Pola tidak cocok kriteria di atas. Butuh data latih lebih banyak. |

> **Catatan GANGGUAN TRANSIEN:** Setiap penyebab transien (PETIR, Layang-Layang, Hewan, Benda Asing) memiliki **karakteristik gelombang yang dapat dibedakan** — durasi, arus puncak, rasio i0/i1, dan jumlah sub-fault menunjukkan pola berbeda di data berlabel yang ada. Keterbatasan saat ini bukan pada kemampuan teknis, melainkan **jumlah data berlabel per kelas yang belum cukup** untuk melatih classifier multi-class yang andal secara statistik (maks. 7 sampel per kelas non-PETIR). Sebagai solusi sementara, sistem menggunakan heuristik berbasis pola yang teramati dari data untuk menampilkan estimasi probabilitas per penyebab. Retraining multi-class akan dilakukan setelah ≥30 sampel berlabel per kelas terkumpul melalui konfirmasi stakeholder.

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
XGBoost binary classifier, dilatih dari **132 file berlabel** (rele jarak yang berhasil ter-parse).

#### Distribusi Data Berlabel

| Kelas | Jumlah Sampel | Status untuk ML |
|---|---|---|
| PETIR | 111 | ✅ Cukup (digunakan untuk training) |
| LAYANG-LAYANG | 7 | ❌ Kurang — perlu min. ~30–50 sampel |
| BENDA ASING | 4 | ❌ Kurang |
| KONDUKTOR | 3 | ❌ Kurang |
| POHON | 3 | ❌ Kurang |
| HEWAN | 2 | ❌ Kurang |
| LAIN-LAIN | 2 | ❌ Kurang |
| **Total** | **132** | |

Karena data non-PETIR tidak cukup untuk multi-class training, model saat ini hanya membedakan **transien (termasuk PETIR) vs. non-transien**. Estimasi per kelas disediakan melalui heuristik, bukan ML.

Features yang digunakan: `fault_duration_ms`, `fault_count`, `i0/i1 ratio`, `voltage_sag`, `peak_current`, `reclose_status`

### Estimasi Penyebab (Heuristik)
Untuk `GANGGUAN TRANSIEN`, sistem menampilkan estimasi probabilitas per penyebab berdasarkan pola yang teramati dari data berlabel yang ada, dikombinasikan dengan pengetahuan domain proteksi sistem tenaga:

| Penyebab | Sinyal pembeda yang teramati |
|---|---|
| PETIR | Durasi pendek (<100ms), arus puncak tinggi, event tunggal |
| Layang-Layang | Fault count ≥ 3 (kontak berulang saat layang berayun), durasi sedang |
| Hewan | Rasio i0/i1 tinggi (kontak 1-fasa), arus sedang, event tunggal |
| Pohon | Durasi panjang (kontak menetap), AR gagal atau lebih lambat |
| Benda Asing | Fault count ≥ 2, durasi bervariasi |

**Ini adalah heuristik berbasis pola data, bukan output model ML** — karena jumlah sampel berlabel per kelas saat ini tidak mencukupi untuk training statistik. Setelah ≥30 sampel per kelas terkumpul, heuristik ini akan digantikan dengan classifier multi-class terlatih.

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
