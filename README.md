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

## 📊 Data & Model

### Dataset
- **Training Data**: 10,000 sampel siswa (generated)
- **Features**: 6 skor RIASEC + 4 nilai akademik + peminatan
- **Target**: 10 rumpun fakultas

### Model Pipeline
```
Input → Feature Engineering → Ensemble Voting → Top 3 Rumpun → Gemini Chatbot
```

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

## 🛠️ Tech Stack

| Kategori | Teknologi |
|----------|-----------|
| **Frontend** | Streamlit, Plotly, Custom CSS |
| **ML** | Scikit-learn, XGBoost, LightGBM |
| **AI** | Google Gemini 2.5 Flash |
| **Database** | PostgreSQL (Supabase) |
| **Deployment** | Streamlit Cloud / Heroku |

---
