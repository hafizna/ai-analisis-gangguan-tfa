# Panduan Penggunaan Pipeline

## 1. Klasifikasi File Tunggal

```bash
cd pipeline/
python models/predict.py path/to/file.cfg
```

Contoh output:
```
Gangguan Saluran Transmisi: GI KUDUS — BAY SAYUNG
Klasifikasi  : GANGGUAN TRANSIEN
Confidence   : 95%
Tier         : 2 (ML Classifier)

Evidence:
  Classifier ML: pola transien terdeteksi (prob=95%)
  dur=79ms  fault_count=1  i0/i1=1.23
  Estimasi penyebab: PETIR 61%  |  Hewan 16%  |  Layang-Layang 12%  |  Benda Asing 8%  |  Pohon 3%
  Konfirmasi via data cuaca atau inspeksi lapangan.

Fault Parameters:
  Duration     : 79 ms
  Fault Count  : 1
  Peak Current : 4821 A
  I0/I1 Ratio  : 1.23
  Voltage Sag  : 0.85 pu
  Zone         : Z1
  Trip Type    : SINGLE_PHASE
  Reclose OK   : True
```

---

## 2. Klasifikasi Batch

```bash
python batch_predict.py
```

- Memindai seluruh `raw_data/` secara rekursif
- Output disimpan ke `data/predictions/all_predictions.csv`
- Error disimpan ke `data/predictions/prediction_errors.csv`
- Menampilkan distribusi prediksi dan akurasi pada file berlabel

### Format Output CSV

| Kolom | Keterangan |
|---|---|
| `predicted_label` | Hasil klasifikasi |
| `confidence` | Kepercayaan model (0-1) |
| `tier` | 1=aturan, 2=ML, 0=fallback |
| `evidence` | Detail analisis + estimasi % penyebab |
| `folder_label` | Label dari nama folder (jika ada) |
| `correct` | **Diisi stakeholder** — apakah prediksi benar? |
| `notes` | **Diisi stakeholder** — catatan lapangan |

---

## 3. Web App

```bash
python webapp/app.py
# Buka http://localhost:5000
```

Fitur:
- Upload file .cfg + .dat
- Tampilkan hasil klasifikasi + estimasi penyebab
- Konfirmasi atau koreksi hasil
- Riwayat analisis tersimpan di `webapp/history.csv`

---

## 4. Melatih Ulang Model

Model saat ini: **XGBoost binary** (transien vs permanen), 132 sampel berlabel.

Untuk retraining setelah ada data berlabel baru:
```bash
python models/train.py
```

Untuk retraining multi-class (butuh ≥30 sampel per kelas):
- Tambahkan label di `data/labels/`
- Modifikasi `models/train.py` untuk multi-class output
- Jalankan ulang training

---

## 5. Kenapa File Gagal Diklasifikasikan?

| Error | Penyebab | Solusi |
|---|---|---|
| `COMTRADE parse failed` | File .dat hilang / korup | Cek kelengkapan file |
| `No fault detected` | Window rekaman tidak mengandung gangguan | Normal — bukan error |
| `Feature extraction failed` | Nama channel tidak dikenali | Tambahkan pola di `config/channel_mappings.json` |
| `DIFFERENTIAL is not supported` | File dari rele 87L | Belum didukung |
| `DIRECTIONAL_EF is not supported` | File dari rele 67N | Belum didukung |

---

## 6. Menambahkan Pola Channel Baru

Edit `config/channel_mappings.json`:

```json
"MERK_BARU": {
  "channel_patterns": {
    "IA": ["pola_A", "pola_lain"],
    "IB": ["pola_B"],
    "IC": ["pola_C"],
    "VA": ["pola_VA"],
    ...
  }
}
```

Tambahkan deteksi merk di `core/channel_normalizer.py` → fungsi `detect_manufacturer`.
