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
| NR Electric PCS900 | `CB1.TrpA/B/C`, `DZ1R/S/T`, `CB1.79.Succ_Rcls` |
| Qualitrol DFR Eksternal | Channel `F21_REC` + format RST |

Normalisasi ini menyelesaikan masalah yang selama ini membuat analisis DFR bergantung pada penanganan manual per rekaman.

**2. Deteksi Proteksi & Ekstraksi Fitur Otomatis**
Dari rekaman mentah, sistem secara otomatis mengidentifikasi:
- Tipe proteksi yang bekerja (rele jarak Z1/Z2/Z3, AR, trip fasa)
- Inception time, durasi, dan jumlah sub-fault
- Fitur kuantitatif: di/dt, arus puncak, rasio i0/i1, voltage sag, impedansi

**3. Klasifikasi dengan Akurasi 95% pada Data Berlabel**
Dari 497 file COMTRADE yang diproses dari 9 UPT:

| Item | Hasil |
|---|---|
| Berhasil diklasifikasikan | 392 file (79%) |
| Tidak dapat diklasifikasikan | 105 file (parse gagal / no fault / proteksi lain) |
| Akurasi pada 132 file berlabel | **95%** (125/132 file tepat) |
| Proteksi yang didukung | Rele Jarak (Distance 21) |
| Proteksi yang belum didukung | Diferensial (87L), Arah EF (67N) |

---

## Alur Klasifikasi (End-to-End)

```
File COMTRADE (.cfg + .dat)
        │
        ▼
┌─────────────────────────────┐
│  COMTRADE Parser             │  IEEE C37.111 (1997 & 1999)
│  Channel Normalizer          │  ABB / Siemens / GE / NR / Qualitrol
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  Protection Router           │  Baca sinyal digital:
│                              │  - Tipe proteksi (21/87L/67N)
│                              │  - Zona trip (Z1/Z2/Z3)
│                              │  - Trip type (single/three pole)
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  Fault Detector              │  Deteksi dari status channels / waveform:
│  Feature Extractor           │  - Inception time, durasi, fault_count
│                              │  - Peak current, di/dt, i0/i1, voltage sag
│                              │  - Reclose outcome (berhasil / gagal)
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────┐
│  KLASIFIKASI — 4 Lapisan (berurutan, first-match wins)   │
│                                                          │
│  Layer 0  reclose_confirmed_transient                    │
│     AR berhasil + arus gangguan nyata (>200A)            │
│     → GANGGUAN TRANSIEN (95% conf)                       │
│                                                          │
│  Layer 1  fault_on_reclose_phase_change  (Tier 1 Rule)   │
│     2 fault events, beda fasa, dur >80ms, AR gagal       │
│     → KONDUKTOR / KERUSAKAN PERALATAN                    │
│                                                          │
│  Layer 1  three_pole_failed_reclose      (Tier 1 Rule)   │
│     Trip 3-fasa, AR gagal, peak >50A                     │
│     → GANGGUAN PERMANEN                                  │
│                                                          │
│  Layer 1  explicit_failed_reclose        (Tier 1 Rule)   │
│     AR gagal, dur >10ms, peak >100A                      │
│     → GANGGUAN PERMANEN                                  │
│                                                          │
│  Layer 2  Decision Tree ML Classifier    (Tier 2 ML)     │
│     Fitur: di/dt, peak_current, i0/i1, voltage_sag       │
│     → GANGGUAN TRANSIEN + estimasi % per penyebab        │
│       (dengan catatan confidence berdasarkan prob ML)    │
│                                                          │
│  Fallback  (Tier 0)                                      │
│     → PERLU INVESTIGASI (jika model tidak dimuat)        │
└─────────────────────────────────────────────────────────┘
             │
             ▼
  Hasil + Confidence + Evidence + Rekomendasi
  + Estimasi % penyebab (PETIR / Layang / Hewan / Benda Asing / Pohon)
```

### Penjelasan Tiap Layer

**Layer 0 — Confirmed Transient Shortcut**
Jika AR (Auto-Reclose) berhasil dan arus gangguan nyata terdeteksi (>200A), gangguan terbukti bersifat transien — jalur ini memotong ML dan langsung mengembalikan GANGGUAN TRANSIEN dengan confidence 95%. Ini menangani mayoritas kasus rekaman bertipe "RECLOSE SUKSES".

**Layer 1 — Aturan Deterministik (Tier 1)**
Tiga aturan berbasis sinyal proteksi:
- *Phase change on reclose*: pola impedansi/fasa berubah antar event = tower roboh / konduktor putus
- *3-phase failed reclose*: trip 3-fasa + AR gagal = gangguan permanen
- *Any failed reclose*: AR gagal dengan arus dan durasi nyata = gangguan permanen

Setiap aturan memiliki guard condition terhadap rekaman dead-time dan artefak deteksi (arus <100A, durasi <10ms) sehingga tidak terpicu pada file rekaman di sisi remote yang tidak mengandung arus gangguan.

**Layer 2 — ML Classifier (Tier 2)**
Decision Tree yang dilatih dari 56 sampel berkualitas (setelah filter Tier 1). Model membedakan pola waveform berdasarkan di/dt, peak current, dan rasio i0/i1. Karena data non-PETIR terbatas, output ML (baik `pred=1` maupun `pred=0`) sama-sama dikembalikan sebagai GANGGUAN TRANSIEN dengan confidence berbeda — disertai estimasi probabilitas per penyebab dari heuristik.

**Estimasi Penyebab (Heuristik)**
Setiap output GANGGUAN TRANSIEN dilengkapi estimasi % untuk 5 penyebab berdasarkan aturan domain knowledge:

| Penyebab | Pola pembeda |
|---|---|
| PETIR | Durasi pendek (<100ms), arus puncak tinggi, event tunggal |
| Layang-Layang | fault_count ≥ 3 (kontak berulang saat layang berayun), durasi sedang |
| Hewan | i0/i1 tinggi (kontak 1-fasa), arus sedang, event tunggal |
| Pohon | Durasi panjang (>300ms), AR lebih lambat atau gagal |
| Benda Asing | fault_count ≥ 2, durasi bervariasi |

---

## Klasifikasi Output

| Label | Keterangan |
|---|---|
| **GANGGUAN TRANSIEN** | Gangguan singkat, AR berhasil atau pola waveform transien. Dilengkapi estimasi % per penyebab. |
| **GANGGUAN PERMANEN** | AR gagal, gangguan persisten. Perlu investigasi jalur. |
| **KONDUKTOR / KERUSAKAN PERALATAN** | Perubahan fasa antar-event saat AR — indikasi tower roboh / konduktor putus. |
| **PERLU INVESTIGASI** | Model tidak tersedia. Jalankan `models/train.py` terlebih dahulu. |

> **Catatan:** PETIR, Layang-Layang, Hewan, dan Benda Asing menghasilkan karakteristik gelombang yang serupa — semua tergolong GANGGUAN TRANSIEN. Estimasi % per penyebab adalah **heuristik berbasis domain knowledge**, bukan output ML. Konfirmasi via data cuaca, CCTV, atau inspeksi lapangan tetap diperlukan. Keterbatasan ini bukan kelemahan teknis sistem, melainkan konsekuensi dari jumlah data berlabel per kelas yang belum mencukupi untuk training multi-class (maks. 7 sampel per kelas non-PETIR saat ini).

---

## Distribusi Data Berlabel

| Kelas | Jumlah Sampel | Status untuk ML |
|---|---|---|
| PETIR | 111 | Digunakan untuk training |
| LAYANG-LAYANG | 7 | Kurang — perlu min. ~30 sampel |
| BENDA ASING | 4 | Kurang |
| KONDUKTOR | 3 | Kurang |
| POHON | 3 | Kurang |
| HEWAN | 2 | Kurang |
| LAIN-LAIN | 2 | Kurang |
| **Total** | **132** | |

---

## Relay yang Didukung

| Merk | Model | Deteksi |
|---|---|---|
| Siemens | 7SA, 7SL (SIPROTEC 4) | Nama sinyal + rec_dev_id |
| Siemens | SIPROTEC 5 (DIGSI 5) | BM-prefix + MPI3/MPV3 channels |
| ABB | REL670, RED670 | rec_dev_id + LINE UL1/UL2/UL3 |
| GE / Multilin | P442, P444, D60 | rec_dev_id + START Z/TRIP ZONE |
| Schneider | P442 | rec_dev_id |
| NR Electric | PCS900 | CB1.TrpA/B/C, DZ1R/S/T, 79.Succ_Rcls |
| Qualitrol | DFR Eksternal | F21_REC channels + RST format |

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
cd pipeline/
python batch_predict.py
# Output: data/predictions/all_predictions.csv
#         data/predictions/prediction_errors.csv
```

### Klasifikasi File Tunggal
```bash
python models/predict.py path/to/file.cfg
```

### Melatih Ulang Model
```bash
# Setelah ada data berlabel baru dari konfirmasi stakeholder
python models/train.py
```

---

## Deploy ke Railway

Railway membaca `Procfile` dan `railway.json` secara otomatis.

```
1. git push origin master
2. railway.app → New Project → Deploy from GitHub
3. Pilih repo ini (hafizna/ai-analisis-gangguan-tfa)
4. Railway auto-detect Nixpacks → install requirements.txt → start gunicorn
5. Settings → Generate Domain → selesai
```

Variabel yang tidak perlu dikonfigurasi manual — app berjalan tanpa env vars tambahan.

---

## Struktur Folder

```
pipeline/
├── core/
│   ├── comtrade_parser.py       # Parse COMTRADE, normalisasi channel
│   ├── channel_normalizer.py    # Mapping nama channel → IA/IB/IC/VA/VB/VC
│   ├── protection_router.py     # Deteksi tipe & zona proteksi dari sinyal digital
│   ├── fault_detector.py        # Deteksi inception, durasi, reclose outcome
│   └── feature_extractor.py    # Ekstraksi fitur kuantitatif untuk ML
├── models/
│   ├── rules.py                 # Aturan deterministik Layer 1 (Tier 1)
│   ├── train.py                 # Training Decision Tree Tier 2
│   ├── predict.py               # Inferensi end-to-end + Layer 0 shortcut
│   └── petir_tree.pkl           # Model terlatih (56 sampel berkualitas)
├── data/
│   ├── predictions/             # Output batch (gitignored)
│   └── features/                # Fitur ekstraksi (gitignored)
├── webapp/
│   ├── app.py                   # Flask web app
│   └── templates/               # Jinja2 templates (index, result, browse, history)
├── batch_predict.py             # Klasifikasi batch seluruh raw_data/
├── requirements.txt
├── Procfile                     # Railway / gunicorn start command
└── railway.json                 # Railway deploy config
```

---

## Roadmap

| Tahap | Status | Keterangan |
|---|---|---|
| Parser COMTRADE multi-merk | Selesai | ABB / Siemens / GE / NR / Qualitrol |
| Deteksi proteksi jarak | Selesai | Zone 1-3, AR, fasa trip, reclose outcome |
| Klasifikasi transien vs permanen | Selesai | 95% akurasi pada 132 file berlabel |
| Estimasi probabilitas per penyebab | Selesai (heuristik) | Menunggu data berlabel cukup untuk ML |
| Web app upload & klasifikasi | Selesai | Deploy ke Railway |
| Retraining multi-class | Menunggu | Butuh ≥30 sampel berlabel per kelas |
| Crosscheck lapangan | Menunggu | Stakeholder isi kolom `correct`/`notes` di all_predictions.csv |
