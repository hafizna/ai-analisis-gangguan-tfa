# AI Analisis Gangguan DFR

Sistem klasifikasi gangguan berbasis COMTRADE IEEE C37.111 untuk rekaman transmisi dan trafo.

Update terakhir: 9 April 2026

Dokumen ini merangkum:
- tahap yang sudah selesai
- kemampuan yang sudah aktif di kode terbaru
- arti `status_data` dan `suspected_label`
- temuan terbaru dari scan data dan parser

---

## Ringkasan Terkini

- Parser COMTRADE sudah lebih tahan terhadap variasi CFG, termasuk kasus `NR,TRANSFORMER_RELAY,1999,` dan tanggal `DD/MM/YYYY`.
- Jalur line/transmisi dan jalur transformer differential sekarang dipisah.
- Browse dan batch prediction sudah path-aware.
- UI waveform sudah mendukung dua cursor yang bisa dipilih user.
- `status_data` dan `suspected_label` adalah heuristik untuk triage, bukan label final.

---

## Tahap Proyek

### Tahap 1 - Fondasi line/transmisi
Selesai.

- Parse `cfg/dat`
- Normalisasi channel multi-merk
- Deteksi proteksi, zona, trip type, dan reclose outcome
- Ekstraksi fitur gangguan
- Klasifikasi line cause: `GANGGUAN TRANSIEN`, `GANGGUAN PERMANEN`, `KONDUKTOR / KERUSAKAN PERALATAN`, `PERLU INVESTIGASI`

### Tahap 2 - Transformer differential
Selesai untuk discovery dan routing, aktif di UI.

- Channel trafo dimapping ke HV/LV/differential/restraint
- Rekaman 87T/PDIF-style bisa terdeteksi
- Halaman transformer history, trends, dan data-status tersedia
- Rekaman trafo yang punya filename generik tetap bisa ditemukan lewat folder/path

### Tahap 3 - Discovery dan rekap data
Selesai.

- Browse tidak lagi bergantung pada nama file saja
- `status_data` memisahkan `TRANSIENT`, `TRANSFORMER`, dan `UNKNOWN`
- `suspected_label` menandai label dugaan dengan awalan `DIDUGA`
- Batch export ikut menulis field ini ke CSV

### Tahap 4 - Data curation
Berjalan.

- Stakeholder masih perlu crosscheck isi COMTRADE dan label lapangan
- File OCR-only tidak boleh dipaksa menjadi `87T CONFIRMED`
- Kelas non-PETIR dan kasus trafo tetap butuh kurasi tambahan

---

## Temuan Terbaru

1. Banyak file trafo memakai nama CFG yang generik.
   Folder/path lebih informatif daripada nama file.

2. Rekaman OCR-only memang ada.
   Kasus seperti ini sekarang dipisah sebagai `OCR ONLY`, bukan dianggap `87T CONFIRMED`.

3. Parser perlu menangani CFG non-standar.
   Kasus `Trafo 1_Diff` sempat gagal karena header `NR,TRANSFORMER_RELAY,1999,` dan format tanggal `DD/MM/YYYY`. Keduanya sudah ditangani.

4. Browse sekarang lebih cocok dipakai untuk triage.
   Hasil scan lokal per 9 April 2026 menemukan 531 pasangan CFG+DAT yang dapat dipetakan:
   - 158 `TRANSIENT`
   - 38 `TRANSFORMER`
   - 335 `UNKNOWN`

   Catatan: `UNKNOWN` berarti tidak ada petunjuk path yang cukup, bukan parse gagal.

5. Path heuristics dipakai sebagai petunjuk, bukan kebenaran final.
   Routing klasifikasi tetap harus mengacu ke isi COMTRADE, bukan nama folder.

---

## Makna Label

| Field | Arti | Dipakai untuk |
| --- | --- | --- |
| `status_data` | `TRANSIENT`, `TRANSFORMER`, atau `UNKNOWN` | Rekap dan filter triage |
| `path_tag` | Contoh: `PETIR`, `LAYANG-LAYANG`, `87T CONFIRMED`, `OCR ONLY`, `TRAFO CANDIDATE` | Badge dan hint di browse |
| `suspected_label` | Label dugaan, misalnya `DIDUGA PETIR` | Tampilan yang tidak bersifat fixed |
| `folder_label` | Label dari nama folder jika tersedia | Crosscheck stakeholder di CSV |

Penting:
- `status_data` bukan hasil klasifikasi akhir.
- `suspected_label` bukan label final.
- Keduanya hanya membantu triage saat browse dan batch export.

---

## Alur Sistem

1. Parse file `cfg/dat`.
2. Normalisasi channel.
3. Tentukan protection type dan fault event.
4. Ekstrak fitur gangguan.
5. Jalankan jalur line atau jalur transformer.
6. Simpan hasil ke web app, history, atau CSV batch.
7. Secara paralel, path heuristics mengisi status triage untuk browse dan batch.

---

## Output Yang Tersedia

### Line / transmisi
- `GANGGUAN TRANSIEN`
- `GANGGUAN PERMANEN`
- `KONDUKTOR / KERUSAKAN PERALATAN`
- `PERLU INVESTIGASI`

### Transformer
- `INRUSH MAGNETISASI`
- `GANGGUAN INTERNAL TRAFO`
- `GANGGUAN EKSTERNAL (THROUGH)`
- `TEGANGAN LEBIH / OVEREKSITASI`
- `KEMUNGKINAN MALOPERATE`
- `LAIN-LAIN`

### Halaman web
- `/browse` -> rekap path-aware dan filter status data
- `/history` -> riwayat analisis line
- `/transformer/history` -> riwayat transformer
- `/transformer/trends` -> tren class transformer
- `/transformer/data-status` -> kesiapan data transformer

---

## Komponen Utama

| File | Fungsi |
| --- | --- |
| `core/comtrade_parser.py` | Parse COMTRADE dan perbaikan format CFG yang beragam |
| `core/channel_normalizer.py` | Normalisasi nama channel vendor-specific |
| `core/protection_router.py` | Deteksi proteksi dan zona trip |
| `core/fault_detector.py` | Deteksi inception, durasi, dan outcome reclose |
| `core/feature_extractor.py` | Ekstraksi fitur line/transmisi |
| `core/transformer_channel_mapper.py` | Pemetaan channel trafo HV/LV/diff/restraint |
| `core/transformer_feature_extractor.py` | Ekstraksi fitur trafo |
| `core/path_heuristics.py` | `status_data`, `path_tag`, dan `suspected_label` |
| `models/predict.py` | Inference end-to-end |
| `models/transformer_classifier.py` | Klasifikasi transformer |
| `webapp/app.py` | Flask app, browse, history, dan endpoint API |
| `batch_predict.py` | Batch scoring seluruh `raw_data/` |

---

## Cara Menjalankan

### Web app lokal
```bash
cd pipeline
pip install -r requirements.txt
python webapp/app.py
# buka http://localhost:5000
```

### Batch prediction
```bash
cd pipeline
python batch_predict.py
# output:
# data/predictions/all_predictions.csv
# data/predictions/prediction_errors.csv
```

### Klasifikasi file tunggal
```bash
cd pipeline
python models/predict.py path/to/file.cfg
```

### Training ulang
```bash
cd pipeline
python models/train.py
```

---

## Struktur Folder

```text
pipeline/
  core/
    comtrade_parser.py
    channel_normalizer.py
    protection_router.py
    fault_detector.py
    feature_extractor.py
    transformer_channel_mapper.py
    transformer_feature_extractor.py
    path_heuristics.py
  models/
    rules.py
    predict.py
    train.py
    transformer_classifier.py
    petir_tree.pkl
  webapp/
    app.py
    templates/
  data/
    predictions/
    features/
  batch_predict.py
  requirements.txt
  Procfile
  railway.json
```

---

## Catatan Operasional

- Folder/path heuristics membantu menemukan kasus trafo yang filename-nya generik.
- OCR-only cases tetap perlu review manual.
- Jika file tidak punya petunjuk path yang cukup, `status_data` akan menjadi `UNKNOWN`.
- Untuk keputusan final, tetap gunakan hasil classifier berbasis isi COMTRADE.

---

## Roadmap

| Tahap | Status | Catatan |
| --- | --- | --- |
| Parser COMTRADE multi-merk | Done | Sudah menangani variasi CFG utama dan kasus trafo tertentu |
| Line fault classifier | Done | Sudah dipakai di web app dan batch |
| Transformer differential support | Done | Sudah masuk pipeline dan dashboard transformer |
| Path-aware discovery | Done | `status_data` dan `suspected_label` aktif |
| Data curation stakeholder | Ongoing | Isi `correct`/`notes` di CSV batch |
| Perluasan data trafo | Ongoing | Banyak file trafo masih OCR-only atau generik |
| Multi-class non-PETIR | Ongoing | Perlu lebih banyak data berlabel per kelas |

---

## Validasi Lokal

Pengujian yang sudah saya jalankan di workspace ini:

- `python -m pytest pipeline/tests/test_path_heuristics.py pipeline/tests/test_parser.py pipeline/tests/test_stage2_context.py -q`
- `python -m pytest pipeline/tests/test_webapp_waveform.py -q`
- `python -m py_compile pipeline/batch_predict.py pipeline/webapp/app.py pipeline/core/path_heuristics.py`
