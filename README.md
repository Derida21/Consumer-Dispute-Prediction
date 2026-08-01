# 🔮 Consumer Dispute Prediction

Aplikasi machine learning untuk memprediksi apakah sebuah keluhan konsumen (*consumer complaint*) terhadap perusahaan jasa keuangan berpotensi berakhir sebagai **sengketa (dispute)** atau tidak, berdasarkan atribut-atribut keluhan yang diterima.

## 📌 Latar Belakang

Perusahaan jasa keuangan menerima ribuan keluhan pelanggan setiap hari. Sebagian keluhan dapat diselesaikan dengan baik, namun sebagian lainnya berakhir menjadi sengketa (*consumer dispute*) yang membutuhkan biaya, waktu, dan sumber daya tambahan untuk ditangani.

Project ini bertujuan membangun model machine learning yang dapat mengidentifikasi kemungkinan terjadinya sengketa sejak keluhan pertama kali diterima, sehingga perusahaan dapat memprioritaskan penanganan kasus berisiko tinggi secara lebih efektif dan efisien.

**Dataset:** [US Consumer Finance Complaints](https://www.kaggle.com/datasets/kaggle/us-consumer-finance-complaints) — ± 555.957 baris data keluhan konsumen historis.

## 🎯 Objective

1. Mengembangkan sistem prediksi sengketa pelanggan untuk membantu perusahaan jasa keuangan mengidentifikasi keluhan berisiko tinggi.
2. Membangun model klasifikasi biner (*disputed* vs *non-disputed*) melalui evaluasi beberapa algoritma machine learning, cross validation, dan hyperparameter tuning.

**Target metrik:** F1-Score ≥ 75% dan ROC-AUC ≥ 0.8 (lihat bagian [Model & Metodologi](#-model--metodologi) untuk hasil aktual).

## ✨ Fitur Aplikasi

Aplikasi Streamlit ini terdiri dari dua halaman:

| Halaman | Deskripsi |
|---|---|
| **📊 EDA** | Eksplorasi data: ringkasan dataset, missing values, top 10 perusahaan dengan keluhan terbanyak, serta analisis hubungan antara `issue`, `state`, `company_response_to_consumer` terhadap status dispute (Chi-Square Test). |
| **🔮 Prediction** | Form input atribut keluhan (produk, issue, perusahaan, respons perusahaan, dll.) untuk memprediksi apakah keluhan tersebut berpotensi menjadi dispute. |

## 🗂️ Struktur Project

```
Consumer-Dispute-Prediction/
├── modeling/
│   ├── archive.zip                          # dataset terkompresi (consumer_complaints.csv)
│   ├── best_model.pkl                       # model terbaik hasil training (pipeline sklearn)
│   ├── P1M2_derida_falahian.ipynb           # notebook EDA, feature engineering, training model
│   └── P1M2_derida_falahian_inf.ipynb       # notebook inference/testing model
├── src/
│   ├── streamlit_app.py                     # entry point aplikasi Streamlit
│   ├── eda.py                                # halaman Exploratory Data Analysis
│   ├── prediction.py                         # halaman form prediksi
│   └── requirements.txt                      # daftar dependency
└── README.md
```

## 🧠 Model & Metodologi

Beberapa algoritma dibandingkan menggunakan 5-Fold Cross Validation dengan metrik F1-Score, karena target (`consumer_disputed?`) bersifat imbalanced (kelas dispute adalah minoritas):

- Decision Tree
- Random Forest
- XGBoost
- SVM

**Decision Tree** terpilih sebagai model terbaik dan selanjutnya di-tuning menggunakan `GridSearchCV`, menghasilkan peningkatan F1-Score dari baseline (0.2514) menjadi 0.3564 setelah tuning, dengan F1-score akhir 0.37 pada kelas dispute di test set. Model final disimpan sebagai sklearn `Pipeline` (preprocessing + model) di `modeling/best_model.pkl`, sehingga proses encoding fitur kategorikal ditangani otomatis saat inference.

Fitur yang digunakan untuk prediksi mencakup: `product`, `sub_product`, `issue`, `sub_issue`, `consumer_complaint_narrative`, `company_public_response`, `company`, `state`, `tags`, `consumer_consent_provided`, `submitted_via`, `company_response_to_consumer`, `timely_response`, `response_days`, `received_month`, `received_week`, `received_day`, dan `month_in_quarter`.

> Detail lengkap proses EDA, feature engineering, pemilihan algoritma, dan evaluasi model dapat dilihat di `modeling/P1M2_derida_falahian.ipynb`.
>
> **Catatan performa:** F1-score model saat ini (0.37) masih di bawah target awal (≥ 75%), yang wajar mengingat tingkat imbalance kelas dispute yang tinggi pada dataset ini. Peluang peningkatan lebih lanjut dapat dieksplorasi melalui teknik penanganan imbalanced data (misal SMOTE, class weighting) atau algoritma lain seperti Gradient Boosting.

## 🚀 Cara Menjalankan

### 1. Clone repository

```bash
git clone https://github.com/Derida21/Consumer-Dispute-Prediction
cd Consumer-Dispute-Prediction
```

### 2. Buat virtual environment & install dependency

```bash
python -m venv .venv
.venv\ripts\activate      

pip install -r src/requirements.txt
```

### 3. Jalankan aplikasi

```bash
streamlit run src/streamlit_app.py
```

Aplikasi akan terbuka otomatis di browser pada `http://localhost:8501`.

## 📦 Requirements

```
streamlit
pandas
numpy
scikit-learn==1.6.1
matplotlib
seaborn
feature-engine
```

## ⚠️ Catatan Teknis

- File dataset (`consumer_complaints.csv`) disimpan dalam bentuk terkompresi (`modeling/archive.zip`) dan akan diekstrak otomatis saat aplikasi pertama kali dijalankan.
- Pastikan file `modeling/best_model.pkl` dan `modeling/archive.zip` tersedia sebelum menjalankan aplikasi, karena keduanya dimuat langsung oleh `eda.py` dan `prediction.py`.
