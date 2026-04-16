# AI Analisis Gangguan DFR — Pipeline

Sistem klasifikasi penyebab gangguan transmisi berbasis COMTRADE IEEE C37.111.

Update terakhir: April 2026

---

## Ringkasan Sistem

Pipeline ini menerima rekaman COMTRADE dari rele jarak, rele diferensial trafo, dan
DFR eksternal (Qualitrol, Toshiba, dll), lalu mengklasifikasikan penyebab fisik gangguan ke
dalam 6 kelas:

| Kelas | Deskripsi |
|---|---|
| **PETIR** | Sambaran petir langsung atau induced overvoltage |
| **LAYANG-LAYANG** | Kontak layang-layang dengan konduktor |
| **POHON / VEGETASI** | Sentuhan pohon/ranting pada ROW |
| **HEWAN / BINATANG** | Kontak hewan (ular, burung, babi, tikus, dll.) |
| **BENDA ASING** | Benda asing non-hayati (aluminium foil, terpal, spanduk, dll.) |
| **KONDUKTOR / TOWER** | Kerusakan konduktor, joint, klem, atau struktur tower |

---

## Alur Klasifikasi

```
INPUT: file .cfg + .dat (COMTRADE)
        │
        ▼
1. Parse COMTRADE  →  Record
2. determine_protection  →  ProtectionType (DISTANCE / 87T / 87L / UNKNOWN)
3. detect_fault          →  FaultEvent (inception time, duration, reclose outcome)
        │
        ├── [87T] → Transformer classifier (inrush, internal fault, through fault, dll.)
        │
        └── [DISTANCE / generic trip / UNKNOWN] → Line classifier
                │
                ▼
        4. extract_distance_features  →  feature dict
        5. Tier 1 rules (rules.py)
           ├── fault_on_reclose_phase_change   → KONDUKTOR / KERUSAKAN PERALATAN (85%)
           ├── three_pole_failed_reclose        → GANGGUAN PERMANEN (75%)
           └── explicit_failed_reclose          → GANGGUAN PERMANEN (90%)
                │ tidak cocok
                ▼
        6. Tier 2 Multi-class ML (RandomForest, 13 fitur)
           → PETIR / LAYANG-LAYANG / POHON / HEWAN / BENDA ASING / KONDUKTOR
           → probabilitas per kelas ditampilkan di UI
                │ model tidak tersedia
                ▼
        7. Fallback → PERLU INVESTIGASI
```

### Catatan DFR Eksternal

Bila file berasal dari DFR eksternal (Qualitrol, Toshiba standalone) tanpa sinyal trip rele:
- `protection_type = UNKNOWN` namun klasifikasi **tetap berjalan** menggunakan fitur gelombang
- Fitur waveform (di/dt, peak-I, THD, inception angle) tidak bergantung pada jenis rele
- Channel status CB (52A, 52B, PMT BUKA) juga dideteksi untuk konfirmasi trip dan reclose
- Evidence panel menampilkan caveat `[DFR EKSTERNAL]` dan rekomendasi mengkonfirmasi dengan rekaman rele jika tersedia

---

## Komponen Utama

| File | Fungsi |
|---|---|
| `core/comtrade_parser.py` | Parse COMTRADE, perbaikan format CFG beragam |
| `core/channel_normalizer.py` | Normalisasi nama channel multi-merk (ABB, Siemens, NARI, GE, Alstom, Toshiba) |
| `core/protection_router.py` | Deteksi tipe proteksi, zona, trip type, reclose outcome |
| `core/fault_detector.py` | Deteksi inception, durasi, fault count, SOE |
| `core/feature_extractor.py` | Ekstraksi 13 fitur line/transmisi |
| `core/transformer_channel_mapper.py` | Pemetaan channel trafo HV/LV/diff/restraint |
| `core/transformer_feature_extractor.py` | Ekstraksi fitur trafo (H2, H5, slope, DC offset) |
| `models/rules.py` | Tier 1: aturan deterministik KONDUKTOR/PERMANEN |
| `models/train.py` | Training RandomForest 6-kelas (input: labeled_features.csv) |
| `models/predict.py` | Inference end-to-end untuk satu file .cfg |
| `models/transformer_classifier.py` | Klasifikasi event trafo berbasis pengetahuan |
| `models/fault_classifier.pkl` | Model terlatih (5-class, ~400 sampel, CV acc 82.9%) |
| `webapp/app.py` | Flask app: upload, browse, history, API |
| `batch_extract.py` | Ekstraksi fitur batch dari seluruh raw_data/ |
| `extract_all.py` | Ekstraksi arsip ZIP/RAR menggunakan 7-Zip |

---

## Cara Menjalankan

### Web app lokal
```bash
cd pipeline
pip install -r requirements.txt
python webapp/app.py
# buka http://localhost:5000
```

### Ekstraksi arsip (ZIP/RAR)
```bash
# Instal 7-Zip terlebih dahulu: https://www.7-zip.org/
python extract_all.py
# --dry-run  : preview tanpa mengekstrak
# --force    : ekstrak ulang meski sudah ada marker
```

### Batch ekstraksi fitur
```bash
python batch_extract.py
# output: data/features/labeled_features.csv
#         data/features/extraction_errors.csv
```

### Training ulang model
```bash
python models/train.py
# membaca: data/features/labeled_features.csv
# output:  models/fault_classifier.pkl
```

### Klasifikasi file tunggal
```bash
python models/predict.py path/to/file.cfg
```

---

## Model Saat Ini

| Parameter | Nilai |
|---|---|
| Algoritma | RandomForestClassifier (300 trees) |
| Kelas | PETIR, LAYANG, POHON, HEWAN, BENDA_ASING, KONDUKTOR |
| Sampel training | ~400 baris (setelah quality filter + Tier 1 exclusion) |
| Fitur | 13 (lihat `models/train.py:FEATURE_COLS`) |
| CV accuracy | 82.9% ± 2.6% (5-fold stratified) |
| CV F1 weighted | 78.3% ± 3.5% |
| Catatan | Kelas POHON terbatas (< 2 sampel usable) — model belum bisa prediksi POHON |

---

## Output Klasifikasi

### Line / transmisi
- `PETIR` / `LAYANG-LAYANG` / `POHON / VEGETASI` / `HEWAN / BINATANG` / `BENDA ASING` / `KONDUKTOR / TOWER`
- `GANGGUAN PERMANEN` (Tier 1 permanent fault rules)
- `KONDUKTOR / KERUSAKAN PERALATAN` (Tier 1 conductor fault rule)
- `PERLU INVESTIGASI` (fallback)

### Transformer differential (87T)
- `INRUSH MAGNETISASI`
- `GANGGUAN INTERNAL TRAFO`
- `GANGGUAN EKSTERNAL (THROUGH)`
- `TEGANGAN LEBIH / OVEREKSITASI`
- `KEMUNGKINAN MALOPERATE`
- `PERLU INVESTIGASI`

---

## Struktur Folder

```text
pipeline/
  core/
    comtrade_parser.py
    channel_normalizer.py
    channel_mappings.json       ← pola channel per merk (NARI, ABB, Siemens, Toshiba, dll.)
    protection_router.py
    fault_detector.py
    feature_extractor.py
    transformer_channel_mapper.py
    transformer_feature_extractor.py
    path_heuristics.py
  models/
    rules.py
    train.py
    predict.py
    transformer_classifier.py
    fault_classifier.pkl        ← model aktif (5-class RandomForest)
    petir_tree.pkl              ← alias legacy (sama dengan fault_classifier.pkl)
  webapp/
    app.py
    templates/
      index.html
      results.html
      history.html
      browse.html
  data/
    features/                   ← di-gitignore; diisi oleh batch_extract.py
  batch_extract.py
  extract_all.py
  requirements.txt
```

---

## Status Pengembangan

| Komponen | Status | Catatan |
|---|---|---|
| Parser COMTRADE multi-merk | Selesai | NARI, ABB, Siemens, GE, Alstom, Toshiba, Qualitrol |
| Deteksi CB status (52A/52B, PMT) | Selesai | Digunakan untuk konfirmasi trip/reclose DFR tanpa sinyal rele |
| Tier 1 rule engine | Selesai | 3 aturan deterministik KONDUKTOR/PERMANEN |
| Multi-class fault classifier | Selesai | 5-class RF, accuracy 82.9%. POHON butuh data lebih |
| Transformer differential support | Selesai | H2/H5/slope/DC offset, 6 kelas event |
| Web app & browse | Selesai | Upload, browse raw_data, history |
| Batch extraction pipeline | Selesai | ZIP/RAR via 7-Zip, skip duplikat, error log |
| Kurasi data stakeholder | Berlanjut | Isi `correct`/`notes` di labeled_features.csv |
| Data kelas POHON | Kurang | Perlu minimal 10+ rekaman berlabel POHON untuk training |
