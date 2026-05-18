# 🎓 RekoJurusan AI

> Sistem rekomendasi jurusan kuliah berbasis AI untuk siswa SMA — Analisis RIASEC + Machine Learning + Chatbot Gemini

[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)

---

## 🚀 Fitur

- **Kuesioner RIASEC** — 42 pertanyaan Holland Code untuk analisis kepribadian
- **Input Nilai** — Dinamis sesuai peminatan (IPA/IPS/Bahasa)
- **Prediksi ML** — Ensemble 4 model (RF, GB, XGB, LGBM) untuk 10 rumpun fakultas
- **250+ Jurusan** — Database lengkap dengan PTN/PTS rekomendasi
- **AI Chatbot** — Konseling interaktif dengan Google Gemini 2.5 Flash
- **Visualisasi** — Radar chart & bar chart dengan Plotly

---

## 📦 Instalasi

```bash
# Clone repo
git clone <repo-url>
cd rekojurusan-ai

# Install dependencies
pip install -r requirements.txt
```

**`requirements.txt`**
```txt
streamlit==1.32.0
pandas==2.2.0
numpy==1.26.0
scikit-learn==1.4.0
xgboost==2.0.3
lightgbm==4.3.0
plotly==5.18.0
google-genai==0.5.0
gdown==5.1.0
```

---

## ⚙️ Konfigurasi

### 1. Gemini API Key
Dapatkan gratis di [Google AI Studio](https://ai.google.dev/) → Masukkan di sidebar saat aplikasi berjalan

### 2. Model ML
Model otomatis download dari Google Drive saat pertama kali run. Jika gagal, download manual:
- **Link**: [Google Drive](https://drive.google.com/file/d/1dM11rQS6D4g92eA59yJ9hFPP3qZlhEaH/view?usp=sharing)
- **Filename**: `ENSEMBLE_VOTING_CLASSIFIER.pkl`
- **Letakkan**: Di folder yang sama dengan `mainappgenai.py`

---

## ▶️ Cara Menjalankan

```bash
streamlit run mainappgenai.py
```

Buka `http://localhost:8501` di browser

---

## 📊 Data & Model

### Dataset
- **Training Data**: 10,000 sampel siswa (generated)
- **Features**: 6 skor RIASEC + 4 nilai akademik + peminatan
- **Target**: 10 rumpun fakultas
- **Database**: PostgreSQL (Supabase) untuk eksplorasi data

### Model Pipeline
```
Input → Feature Engineering → Ensemble Voting → Top 3 Rumpun → Gemini Chatbot
```

**Model Details:**
- Random Forest (100 trees)
- Gradient Boosting (100 estimators)
- XGBoost (100 estimators)
- LightGBM (100 estimators)

**Akurasi**: ~75% (top-1), ~90% (top-3)

### 10 Rumpun Fakultas
1. Teknik & Informatika
2. Kedokteran & Kesehatan
3. Ekonomi & Bisnis
4. Sains & Matematika
5. Sosial & Humaniora
6. Hukum & Politik
7. Pendidikan
8. Seni & Desain
9. Pertanian & Lingkungan
10. Komunikasi & Media

---

## 📁 Struktur Proyek

```
rekojurusan-ai/
├── mainappgenai.py                # Aplikasi utama
├── train_model.ipynb              # Training ML model
├── SQL_Data_Exploration.ipynb     # Eksplorasi data dengan SQL
├── ENSEMBLE_VOTING_CLASSIFIER.pkl # Model terlatih (auto-download)
├── requirements.txt
└── README.md
```

---

## 🛠️ Tech Stack

| Kategori | Teknologi |
|----------|-----------|
| **Frontend** | Streamlit, Plotly, Custom CSS |
| **ML** | Scikit-learn, XGBoost, LightGBM |
| **AI** | Google Gemini 2.5 Flash |
| **Database** | PostgreSQL (Supabase) |
| **Deployment** | Streamlit Cloud / Heroku |

---

## 🔧 Troubleshooting

**Error: Module not found**
```bash
pip install <module-name> --upgrade
```

**Error: Gemini API quota habis**
- Free tier: 60 request/menit
- Solusi: Tunggu atau upgrade ke paid tier

**Model tidak terdownload**
- Download manual dari link di atas
- Pastikan nama file exact: `ENSEMBLE_VOTING_CLASSIFIER.pkl`

---

## 🚀 Deploy ke Cloud

**Streamlit Cloud (Gratis):**
1. Push ke GitHub
2. Deploy di [share.streamlit.io](https://share.streamlit.io/)
3. Tambahkan `GEMINI_API_KEY` di Secrets

**Heroku:**
```bash
# Procfile
web: streamlit run mainappgenai.py --server.port=$PORT
```

---

## 📝 Lisensi

MIT License — Free to use & modify

---

## 👨‍💻 Developer

Dibuat dengan ❤️ untuk membantu siswa Indonesia memilih jurusan yang tepat

**Kontak:**  
📧 Email: support@rekojurusan.ai (contoh)  
🐛 Issues: [GitHub Issues](https://github.com/username/repo/issues)

---

⭐ **Star repo ini jika membantu!**
