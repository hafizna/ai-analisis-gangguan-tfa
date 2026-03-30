# Panduan Penggunaan Pipeline

## 1. Klasifikasi File Tunggal

```bash
cd pipeline/
python models/predict.py path/to/file.cfg
```

Contoh output:
```
============================================================
  File    : PCS900_RCD_01016_20240518_044955_016.CFG
  Label   : GANGGUAN TRANSIEN
  Tier    : 1  (reclose_confirmed_transient)
  Conf.   : 95%
  Evidence: AR berhasil — gangguan transien terkonfirmasi.
            peak_i=953A  dur=57ms  fault_count=1
            Estimasi penyebab: PETIR 51% | Hewan 30% | Layang-Layang 10% | ...
============================================================
  Station      : NR
  Relay        : LINE_DISTANCE_RELAY
  Zone         : Z1
  Trip type    : single_pole
  Phases       : C
  Duration     : 57 ms
  fault_count  : 1
  peak_I       : 953 A
  i0/i1        : 1.227
  voltage sag  : 0.107 pu
  Reclose ok   : True
```

---

## 2. Klasifikasi Batch

```bash
python batch_predict.py
```

- Memindai seluruh `raw_data/` secara rekursif
- Melewati folder `olah/`, `_extracted/`, `locus/`, `analisa/`
- Output disimpan ke `data/predictions/all_predictions.csv`
- Error disimpan ke `data/predictions/prediction_errors.csv`
- Menampilkan distribusi prediksi dan akurasi pada file berlabel

### Format Output CSV

| Kolom | Keterangan |
|---|---|
| `predicted_label` | Hasil klasifikasi |
| `confidence` | Kepercayaan model (0–1) |
| `tier` | 1 = aturan/shortcut, 2 = ML, 0 = fallback |
| `rule_name` | Nama layer yang terpicu |
| `evidence` | Detail analisis + estimasi % penyebab |
| `folder_label` | Label dari nama folder (jika ada) |
| `correct` | **Diisi stakeholder** — apakah prediksi benar? |
| `notes` | **Diisi stakeholder** — catatan lapangan |

---

## 3. Web App

```bash
pip install -r requirements.txt
python webapp/app.py
# Buka http://localhost:5000
```

Fitur:
- Upload file `.cfg` + `.dat` atau pilih dari `raw_data/` via Browse
- Tampilkan hasil klasifikasi + evidence + estimasi % penyebab
- Rekomendasi tindak lanjut berdasarkan penyebab tertinggi
- Dark mode toggle
- Riwayat analisis tersimpan di `webapp/history.csv`

---

## 4. Urutan Layer Klasifikasi

Sistem menjalankan layer secara berurutan. Layer pertama yang cocok mengembalikan hasil final — layer berikutnya tidak dijalankan.

```
INPUT: fault event terdeteksi
        │
        ▼
Layer 0: reclose_confirmed_transient
  Syarat: reclose_successful=True AND peak_current > 200A
  → GANGGUAN TRANSIEN (95% conf) + estimasi penyebab
  (Menangani kasus "RECLOSE SUKSES" tanpa perlu ML)
        │ tidak cocok
        ▼
Layer 1a: fault_on_reclose_phase_change
  Syarat: fault_count 2–20, fasa berbeda, dur >80ms, AR tidak berhasil
  → KONDUKTOR / KERUSAKAN PERALATAN (85%)
        │ tidak cocok
        ▼
Layer 1b: three_pole_failed_reclose
  Syarat: trip 3-fasa, AR gagal, peak >50A
  → GANGGUAN PERMANEN (75%)
        │ tidak cocok
        ▼
Layer 1c: explicit_failed_reclose
  Syarat: AR gagal, dur >10ms, peak >100A
  → GANGGUAN PERMANEN (90%)
        │ tidak cocok
        ▼
Layer 2: Decision Tree ML
  Fitur: di/dt, peak_current, i0/i1, voltage_sag, fault_duration, fault_count
  pred=1 (transien): GANGGUAN TRANSIEN + estimasi penyebab
  pred=0 (tidak khas): GANGGUAN TRANSIEN (conf lebih rendah) + estimasi penyebab
  (pred=0 dikembalikan sebagai transien karena data non-PETIR terbatas)
```

---

## 5. Melatih Ulang Model

Model saat ini: **Decision Tree binary** (transien vs tidak khas), 56 sampel berkualitas.

```bash
python models/train.py
```

Proses:
1. Membaca `data/features/labeled_features.csv`
2. Filter sampel berkualitas (`scaling_ok=True`, `duration_ok=True`)
3. Exclude sampel yang sudah ditangani Layer 0/1 (tidak perlu ML)
4. Training Decision Tree dengan `class_weight="balanced"`
5. Output: `models/petir_tree.pkl`

Untuk upgrade ke multi-class setelah data cukup (≥30 per kelas):
- Tambahkan konfirmasi label dari stakeholder di `data/labels/`
- Modifikasi target variabel di `models/train.py`
- Jalankan ulang training

---

## 6. Kenapa File Gagal Diklasifikasikan?

| Error | Penyebab | Solusi |
|---|---|---|
| `COMTRADE parse failed` | File `.dat` hilang / format tidak standar | Cek kelengkapan file |
| `No fault detected` | Window rekaman tidak mengandung gangguan | Normal — bukan error, rekaman dead-time |
| `Feature extraction failed` | Channel arus/tegangan tidak dikenali | Tambahkan pola di `core/channel_normalizer.py` |
| `DIFFERENTIAL is not supported` | File dari rele 87L | Belum didukung |
| `DIRECTIONAL_EF is not supported` | File dari rele 67N | Belum didukung |
| `too many values to unpack` | Format CFG non-standar | Tidak didukung oleh library comtrade |

---

## 7. Menambahkan Pola Channel Baru

Jika merk relay baru menghasilkan nama channel yang tidak dikenali, tambahkan di `core/channel_normalizer.py` — fungsi `detect_manufacturer` dan mapping channel di bagian yang sesuai.

Contoh log warning yang menandakan channel tidak dikenali:
```
Could not normalize channel 'IaR' (unit: A, mfr: UNKNOWN)
```

Cek nama channel di file CFG, lalu tambahkan pattern yang sesuai ke fungsi normalisasi.
