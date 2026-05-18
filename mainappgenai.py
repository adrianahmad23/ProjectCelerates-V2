"""
RekoJurusan AI v2 — Streamlit App
• 42 pertanyaan RIASEC (Ya/Tidak)
• Analisis kepribadian via Google Gemini LLM
• Prediksi Rumpun Fakultas (ML Ensemble)
• Rekomendasi jurusan per rumpun
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle, os, io, time
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
import warnings
warnings.filterwarnings("ignore")

try:
    from google import genai
    GENAI_OK = True
except ImportError:
    GENAI_OK = False

try:
    import gdown
    GDOWN_OK = True
except ImportError:
    GDOWN_OK = False

# ══════════════════════════════════════════════════════════════
#  PAGE CONFIG
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="RekoJurusan AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
* { font-family: 'Plus Jakarta Sans', sans-serif !important; }

.hero {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #312e81 100%);
    padding: 2rem 2rem; border-radius: 16px; margin-bottom: 2rem;
    text-align: center;
}
.hero h1 { color: white; font-size: 2rem; font-weight: 800; margin: 0; }
.hero p  { color: rgba(255,255,255,0.7); font-size: 0.95rem; margin: 0.4rem 0 0; }

.step-badge {
    display: inline-flex; align-items: center; gap: 0.4rem;
    background: #4f46e5;
    color: white; padding: 0.5rem 1.2rem; border-radius: 8px;
    font-weight: 600; font-size: 0.85rem; margin-bottom: 1.5rem;
}

.chat-bubble {
    background: #1e1b4b;
    color: white; border-radius: 14px;
    padding: 1.5rem 1.8rem; margin: 1rem 0;
    line-height: 1.7;
}
.chat-bubble .type-code {
    font-size: 2.8rem; font-weight: 900; letter-spacing: 0.15em;
    background: linear-gradient(90deg, #f59e0b, #ec4899, #8b5cf6);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    display: block; margin: 0.5rem 0;
}
.chat-bubble .type-name { font-size: 1rem; color: #c7d2fe; margin-bottom: 1rem; }

.personality-box {
    background: #1e293b;
    border-left: 4px solid;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin: 0.7rem 0;
    color: #e2e8f0;
}
.personality-box .box-title {
    font-weight: 700; font-size: 1rem; margin-bottom: 0.6rem;
}
.personality-box .box-body {
    color: #cbd5e1; line-height: 1.7; font-size: 0.88rem;
}

.rumpun-card {
    background: #1e293b;
    border-radius: 12px; padding: 1.2rem; margin-bottom: 0.8rem;
    border-left: 4px solid;
}
.rank-1 { border-color: #f59e0b; }
.rank-2 { border-color: #0ea5e9; }
.rank-3 { border-color: #22c55e; }

.metric-box {
    background: #f8f9ff;
    border: 1px solid #e0e7ff; border-radius: 12px;
    padding: 1rem; text-align: center;
}
.metric-box .val { font-size: 1.8rem; font-weight: 800; color: #4f46e5; }
.metric-box .lbl { font-size: 0.78rem; color: #6b7280; margin-top: 0.2rem; }

.jurusan-pill {
    display: inline-block; background: #f1f5f9; border: 1px solid #e0e7ff;
    border-radius: 6px; padding: 0.25rem 0.7rem; margin: 0.2rem;
    font-size: 0.78rem; font-weight: 600; color: #4f46e5;
}

footer { display: none !important; }
.stDeployButton { display: none !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  42 PERTANYAAN RIASEC  (7 per dimensi)
# ══════════════════════════════════════════════════════════════
# Format: (teks pertanyaan, kode_dimensi)
RIASEC_QUESTIONS = [
    # ── R: Realistic (7) ──────────────────────────────────────
    ("Saya suka bekerja dengan kendaraan atau mesin",            "R"),
    ("Saya suka bekerja di luar ruangan",                        "R"),
    ("Saya suka membangun atau merakit sesuatu",                 "R"),
    ("Saya suka merawat hewan",                                  "R"),
    ("Saya orang yang praktis dan suka pekerjaan fisik",        "R"),
    ("Saya suka memperbaiki benda yang rusak",                   "R"),
    ("Saya suka berkebun atau bertani",                          "R"),
    # ── I: Investigative (7) ──────────────────────────────────
    ("Saya suka mengerjakan teka-teki atau soal logika",         "I"),
    ("Saya memiliki kemampuan matematika yang bagus",            "I"),
    ("Saya suka melakukan eksperimen dan penelitian",            "I"),
    ("Saya suka mencari tahu bagaimana segala sesuatu bekerja",  "I"),
    ("Saya menyukai sains dan ilmu pengetahuan",                 "I"),
    ("Saya suka menganalisis masalah secara mendalam",           "I"),
    ("Saya suka membaca buku atau artikel ilmiah",               "I"),
    # ── A: Artistic (7) ───────────────────────────────────────
    ("Saya suka bermain alat musik atau menyanyi",               "A"),
    ("Saya suka bermain peran atau akting dalam drama",          "A"),
    ("Saya suka menggambar, melukis, atau membuat karya seni",   "A"),
    ("Saya menikmati penulisan kreatif (cerpen, puisi, dll.)",   "A"),
    ("Saya adalah orang yang kreatif dan imajinatif",            "A"),
    ("Saya tertarik dengan desain, fashion, atau estetika",      "A"),
    ("Saya suka menonton film, teater, atau pertunjukan seni",   "A"),
    # ── S: Social (7) ─────────────────────────────────────────
    ("Saya tertarik untuk membantu atau menyembuhkan orang",     "S"),
    ("Saya senang belajar tentang budaya dan orang lain",        "S"),
    ("Saya suka bekerja dalam tim dan berkolaborasi",            "S"),
    ("Saya suka membantu orang menyelesaikan masalah mereka",    "S"),
    ("Saya suka mengajar atau melatih orang lain",               "S"),
    ("Saya peduli dengan masalah sosial di masyarakat",          "S"),
    ("Saya suka berdiskusi dan bertukar pikiran dengan orang",   "S"),
    # ── E: Enterprising (7) ───────────────────────────────────
    ("Saya suka menjual atau memasarkan sesuatu",                "E"),
    ("Saya bisa cepat mengambil tanggung jawab baru",            "E"),
    ("Saya ingin memulai bisnis atau usaha sendiri",             "E"),
    ("Saya suka memimpin dan mengarahkan orang lain",            "E"),
    ("Saya orang yang ambisius dan selalu menetapkan tujuan",    "E"),
    ("Saya suka berpidato atau presentasi di depan orang",       "E"),
    ("Saya suka mempengaruhi atau membujuk orang",               "E"),
    # ── C: Conventional (7) ───────────────────────────────────
    ("Saya suka bekerja dengan angka, data, atau tabel",         "C"),
    ("Saya suka mengikuti prosedur dan instruksi yang jelas",    "C"),
    ("Saya tidak keberatan bekerja 8 jam di dalam kantor",       "C"),
    ("Saya memperhatikan detail dan ketelitian",                 "C"),
    ("Saya suka melakukan pengarsipan atau pencatatan",          "C"),
    ("Saya pandai menyimpan dan mengorganisir catatan",          "C"),
    ("Saya suka mengatur berkas, meja, atau dokumen",            "C"),
]

assert len(RIASEC_QUESTIONS) == 42

# ── Deskripsi kepribadian RIASEC (panjang) ───────────────────
RIASEC_FULL = {
    "R": {
        "nama":   "Realistic",
        "alias":  "Si Praktis",
        "warna":  "#ef4444",
        "emoji":  "🔧",
        "singkat": "Bekerja dengan tangan, alat, dan hal-hal nyata",
        "deskripsi": (
            "Orang dengan tipe <b>Realistic</b> memiliki cara berpikir yang praktis. "
            "Dalam memberikan informasi, gayamu cenderung apa adanya, jujur, dan tidak bertele-tele. "
            "Kamu juga mencoba untuk se-spesifik mungkin dalam memberikan informasi, kalau bisa, "
            "disertai contoh gambaran praktis dari penjelasan tersebut.<br><br>"
            "Kamu lebih suka belajar dengan cara mencoba-coba sesuatu, atau dengan mempraktekkan "
            "langsung apa yang kamu pelajari. Kebanyakan orang dengan tipe ini juga lebih suka "
            "bekerja di luar ruangan.<br><br>"
            "Kamu menyukai pekerjaan yang membutuhkan keterampilan fisik, seperti mengoperasikan "
            "benda atau atletik. Hal-hal yang bersifat materi memang lebih menarik dibandingkan "
            "hal-hal yang abstrak dan teoritis. Sehingga, kamu cocok bekerja di tempat yang "
            "mengandalkan kompetensi teknis dan mampu menghasilkan sesuatu yang nyata (terlihat)."
        ),
    },
    "I": {
        "nama":   "Investigative",
        "alias":  "Si Peneliti",
        "warna":  "#3b82f6",
        "emoji":  "🔬",
        "singkat": "Analitis, suka riset, ilmu pengetahuan",
        "deskripsi": (
            "Orang dengan tipe <b>Investigative</b> adalah pengamat yang baik dan suka mencari "
            "pemecahan masalah. Konsep abstrak-teoritis dan permasalahan yang kompleks merupakan "
            "hal yang menyenangkan bagimu. Kamu penuh rasa ingin tahu dan kritis terhadap berbagai "
            "informasi. Meski kritis, kamu tetap terbuka menerima semua informasi karena kamu suka "
            "mempelajari sesuatu.<br><br>"
            "Kamu dengan tipe investigative suka mengamati sesuatu dan tenggelam dalam aktivitas "
            "bidang keilmuan. Kamu suka dengan lingkungan kerja yang mampu mengembangkan kemampuan "
            "berpikirmu dan berfokus pada mencari solusi yang kreatif dan berbasis data."
        ),
    },
    "A": {
        "nama":   "Artistic",
        "alias":  "Si Kreatif",
        "warna":  "#8b5cf6",
        "emoji":  "🎨",
        "singkat": "Kreatif, ekspresif, suka seni dan budaya",
        "deskripsi": (
            "Sebagai orang dengan tipe <b>Artistic</b>, kamu adalah orang yang suka dengan kegiatan "
            "kreatif, seperti penciptaan seni, menulis, produksi film, dan bermain peran. Ketika ada "
            "kesempatan untuk bisa mengekspresikan diri melalui sesuatu yang kreatif, maka kamulah "
            "yang akan tertarik. Estetika adalah hal yang menarik bagimu. Kamu juga suka "
            "berimajinasi. Orang lain bisa melihatmu sebagai sosok yang orisinil karena cara "
            "berpikirmu yang kreatif dan unik.<br><br>"
            "Kamu adalah orang yang fleksibel, bahkan cenderung tertarik dengan hal-hal yang "
            "ambigu. Kamu menyukai lingkungan kerja yang membebaskanmu untuk berkreasi dan "
            "menghargai orisinalitasmu."
        ),
    },
    "S": {
        "nama":   "Social",
        "alias":  "Si Penolong",
        "warna":  "#10b981",
        "emoji":  "🤝",
        "singkat": "Suka membantu, mengajar, dan berkolaborasi",
        "deskripsi": (
            "Orang dengan tipe <b>Social</b> sangat menikmati berinteraksi dengan orang lain, "
            "terutama dalam konteks membantu, mengajarkan, atau mendukung orang lain. Kamu peka "
            "terhadap perasaan dan kebutuhan orang di sekitarmu.<br><br>"
            "Kamu cocok bekerja di lingkungan yang melibatkan banyak interaksi manusiawi, seperti "
            "pendidikan, layanan kesehatan, konseling, atau kegiatan sosial. Kamu percaya bahwa "
            "kolaborasi dan empati adalah kunci untuk menyelesaikan masalah dengan baik.<br><br>"
            "Orang-orang sering datang padamu untuk meminta nasihat atau dukungan emosional, "
            "dan kamu menikmati peran tersebut."
        ),
    },
    "E": {
        "nama":   "Enterprising",
        "alias":  "Si Pemimpin",
        "warna":  "#f59e0b",
        "emoji":  "💡",
        "singkat": "Memimpin, berbisnis, persuasif, ambisius",
        "deskripsi": (
            "Orang dengan tipe <b>Enterprising</b> menyukai tantangan dalam memimpin dan "
            "mempengaruhi orang lain. Kamu adalah orang yang penuh energi, persuasif, dan "
            "berorientasi pada tujuan. Kamu pandai meyakinkan orang lain dan sering menjadi "
            "inisiator dalam kelompok.<br><br>"
            "Kamu tertarik pada dunia bisnis, kewirausahaan, politik, atau manajemen. Kamu "
            "menikmati kompetisi dan tidak takut mengambil risiko untuk meraih tujuan. "
            "Lingkungan kerja yang dinamis dan penuh peluang adalah tempat kamu bisa berkembang "
            "dengan optimal."
        ),
    },
    "C": {
        "nama":   "Conventional",
        "alias":  "Si Terstruktur",
        "warna":  "#6366f1",
        "emoji":  "📊",
        "singkat": "Terstruktur, detail, berorientasi data",
        "deskripsi": (
            "Orang dengan tipe <b>Conventional</b> menyukai keteraturan, struktur, dan konsistensi. "
            "Kamu sangat teliti, rapi, dan dapat diandalkan dalam pekerjaan yang membutuhkan "
            "ketepatan dan kepatuhan terhadap prosedur.<br><br>"
            "Kamu nyaman bekerja dengan angka, data, dokumen, dan sistem yang terorganisasi. "
            "Kamu menikmati lingkungan kerja yang jelas ekspektasinya dan memiliki aturan yang "
            "terstruktur. Pekerjaan seperti akuntansi, administrasi, keuangan, atau analisis "
            "data sangat cocok untukmu."
        ),
    },
}

# ── Rumpun info & jurusan di dalamnya ────────────────────────
RUMPUN_INFO = {
    "Teknik & Informatika": {
        "emoji": "💻",
        "warna": "#4f46e5",
        "deskripsi": "Bidang rekayasa, komputasi, dan teknologi",
        "jurusan": [
            "Teknik Informatika","Ilmu Komputer","Data Science","Kecerdasan Buatan",
            "Rekayasa Perangkat Lunak","Cyber Security","Sistem Informasi",
            "Teknik Elektro","Teknik Mesin","Teknik Sipil","Teknik Kimia",
            "Teknik Industri","Arsitektur","Teknik Perminyakan",
        ],
        "karir": "Software Engineer, Data Scientist, Civil Engineer, Arsitek, AI Engineer",
        "riasec": ["R","I","C"],
    },
    "Sains & MIPA": {
        "emoji": "🔬",
        "warna": "#0ea5e9",
        "deskripsi": "Ilmu alam murni, matematika, dan penelitian",
        "jurusan": [
            "Matematika","Statistika","Aktuaria","Fisika","Kimia",
            "Biologi","Bioteknologi","Astronomi","Geofisika","Oseanografi",
        ],
        "karir": "Peneliti, Aktuaris, Data Analyst, Ilmuwan, Akademisi",
        "riasec": ["I","C","R"],
    },
    "Kesehatan & Kedokteran": {
        "emoji": "🏥",
        "warna": "#ec4899",
        "deskripsi": "Ilmu kesehatan, kedokteran, dan farmasi",
        "jurusan": [
            "Kedokteran","Kedokteran Gigi","Farmasi","Keperawatan",
            "Kebidanan","Kesehatan Masyarakat","Gizi","Kedokteran Hewan",
            "Fisioterapi","Analis Kesehatan",
        ],
        "karir": "Dokter, Apoteker, Perawat, Ahli Gizi, Fisioterapis",
        "riasec": ["I","S","R"],
    },
    "Ekonomi & Bisnis": {
        "emoji": "📈",
        "warna": "#f59e0b",
        "deskripsi": "Bisnis, keuangan, manajemen, dan wirausaha",
        "jurusan": [
            "Akuntansi","Manajemen","Ilmu Ekonomi","Keuangan",
            "Perbankan & Keuangan","Manajemen Pemasaran","Bisnis Digital","Manajemen SDM",
        ],
        "karir": "Akuntan, Manajer, Analis Keuangan, Entrepreneur, HR Manager",
        "riasec": ["E","C","S"],
    },
    "Hukum & Ilmu Sosial": {
        "emoji": "⚖️",
        "warna": "#6366f1",
        "deskripsi": "Hukum, politik, sosial, dan psikologi",
        "jurusan": [
            "Hukum","Ilmu Politik","Hubungan Internasional","Sosiologi",
            "Antropologi","Psikologi","Administrasi Publik","Kriminologi",
        ],
        "karir": "Pengacara, Psikolog, Diplomat, Peneliti Sosial, Konsultan",
        "riasec": ["S","E","I"],
    },
    "Pendidikan & Keguruan": {
        "emoji": "📚",
        "warna": "#10b981",
        "deskripsi": "Ilmu pendidikan, keguruan, dan konseling",
        "jurusan": [
            "Pendidikan Guru SD","Pendidikan Anak Usia Dini","Pendidikan Matematika",
            "Pendidikan Fisika","Pendidikan Bahasa Inggris","Bimbingan Konseling",
            "Pendidikan Biologi","Pendidikan Bahasa Indonesia",
        ],
        "karir": "Guru, Dosen, Konselor, Pengembang Kurikulum, Trainer",
        "riasec": ["S","A","C"],
    },
    "Komunikasi & Media": {
        "emoji": "📡",
        "warna": "#f97316",
        "deskripsi": "Jurnalistik, komunikasi, dan pariwisata",
        "jurusan": [
            "Ilmu Komunikasi","Jurnalistik","Public Relations",
            "Penyiaran (Broadcasting)","Periklanan","Pariwisata","Perhotelan",
        ],
        "karir": "Jurnalis, PR Manager, Content Creator, Event Organizer, Tour Guide",
        "riasec": ["E","A","S"],
    },
    "Seni & Desain": {
        "emoji": "🎨",
        "warna": "#8b5cf6",
        "deskripsi": "Seni rupa, desain, musik, dan film",
        "jurusan": [
            "Desain Komunikasi Visual","Desain Grafis","Animasi",
            "Seni Rupa","Seni Musik","Film dan Televisi","Fotografi","Seni Tari",
        ],
        "karir": "Desainer, Animator, Musisi, Sineas, Fotografer, Ilustrator",
        "riasec": ["A","I","R"],
    },
    "Bahasa & Sastra": {
        "emoji": "🌐",
        "warna": "#14b8a6",
        "deskripsi": "Linguistik, sastra, dan budaya",
        "jurusan": [
            "Sastra Inggris","Sastra Indonesia","Sastra Jepang",
            "Sastra Arab","Sastra Mandarin","Linguistik","Ilmu Perpustakaan",
        ],
        "karir": "Penerjemah, Penulis, Dosen, Jurnalis, Content Writer, Diplomat",
        "riasec": ["A","S","I"],
    },
    "Pertanian & Lingkungan": {
        "emoji": "🌿",
        "warna": "#22c55e",
        "deskripsi": "Pertanian, kehutanan, pangan, dan lingkungan",
        "jurusan": [
            "Agroteknologi","Agribisnis","Peternakan","Perikanan",
            "Kehutanan","Teknologi Pangan","Ilmu Tanah",
        ],
        "karir": "Agronomis, Penyuluh Pertanian, Ahli Pangan, Peneliti Lingkungan",
        "riasec": ["R","I","S"],
    },
}

MAPEL_ALL = ["matematika","fisika","kimia","biologi","bahasa_indonesia",
             "bahasa_inggris","ekonomi","sosiologi","sejarah","seni_budaya"]
MAPEL_LABEL = {
    "matematika":"Matematika","fisika":"Fisika","kimia":"Kimia",
    "biologi":"Biologi","bahasa_indonesia":"B. Indonesia",
    "bahasa_inggris":"B. Inggris","ekonomi":"Ekonomi",
    "sosiologi":"Sosiologi","sejarah":"Sejarah","seni_budaya":"Seni Budaya",
}
NILAI_LABELS = ["Matematika","Fisika","Kimia","Biologi","B.Indo",
                "B.Ing","Ekonomi","Sosiologi","Sejarah","Seni"]

FEATURE_NAMES = [
    'matematika','fisika','kimia','biologi',
    'bahasa_indonesia','bahasa_inggris',
    'ekonomi','sosiologi','sejarah','seni_budaya',
    'riasec_R','riasec_I','riasec_A','riasec_S','riasec_E','riasec_C',
    'sains_avg','sosial_avg','bahasa_avg','kreatif','logika',
    'peminatan_enc','dom_riasec_enc','riasec_dom_gap',
    'sains_vs_sosial','sains_vs_bahasa',
    'bio_vs_fis','kim_vs_mtk','bing_vs_bind',
    'top_sains_idx','top_sosial_idx',
    'riasec_R_I_diff','riasec_A_norm',
]


# ══════════════════════════════════════════════════════════════
#  MODEL LOADING
# ══════════════════════════════════════════════════════════════

# ── ID file dari Google Drive ─────────────────────────────────
GDRIVE_FILES = {
    "preprocessor.pkl":   "1rN2Mtf36IKpY40MXx-dfnaQ4xuZsiwfz",
    "model_xgb.pkl":      "1Fdvo59N2hjPqIV99ttBnsH2haJWCi1rj",
    "model_lgbm.pkl":     "13PPqdWGmaaQFDzRN-D8uXnANrwmb4lkw",
    "model_rf.pkl":       "1peOXgtZ_zW4N_8lZyFxuLe4_oF4sY4Ef",
    "shap_explainer.pkl": "1z4j_5eOloqxvQWoBFSQKNX8f4xco6qQT",
}

def download_models():
    """Download .pkl dari Google Drive kalau belum ada atau invalid."""
    if not GDOWN_OK:
        raise ImportError("Library `gdown` tidak terinstall. Jalankan: pip install gdown")

    for filename, file_id in GDRIVE_FILES.items():
        needs_download = False
        if not os.path.exists(filename):
            needs_download = True
        else:
            # Cek apakah file valid (bukan HTML error page dari GDrive)
            try:
                with open(filename, "rb") as f:
                    header = f.read(5)
                if header[:5] in (b"<!DOC", b"<html", b"<HTML"):
                    os.remove(filename)
                    needs_download = True
                elif os.path.getsize(filename) < 1000:
                    os.remove(filename)
                    needs_download = True
            except Exception:
                needs_download = True

        if needs_download:
            with st.spinner(f"⬇️ Mengunduh {filename}..."):
                url = f"https://drive.google.com/uc?id={file_id}&export=download&confirm=t"
                result = gdown.download(url, filename, quiet=False)
                if result is None:
                    # Fallback ke direct URL
                    gdown.download(
                        f"https://drive.google.com/uc?id={file_id}",
                        filename, quiet=False)
                # Validasi ulang setelah download
                if os.path.exists(filename):
                    with open(filename, "rb") as f:
                        header = f.read(5)
                    if header[:5] in (b"<!DOC", b"<html", b"<HTML"):
                        os.remove(filename)
                        raise ValueError(
                            f"Gagal mengunduh {filename} — Google Drive mengembalikan HTML "
                            f"(file mungkin butuh izin akses publik atau kuota habis)")

@st.cache_resource
def load_models():
    download_models()
    with open("preprocessor.pkl",  "rb") as f: prep = pickle.load(f)
    with open("model_xgb.pkl",     "rb") as f: xgb  = pickle.load(f)
    with open("model_lgbm.pkl",    "rb") as f: lgbm = pickle.load(f)
    with open("model_rf.pkl",      "rb") as f: rf   = pickle.load(f)
    with open("shap_explainer.pkl","rb") as f: shap = pickle.load(f)
    return prep, xgb, lgbm, rf, shap

try:
    prep, xgb_model, lgbm_model, rf_model, shap_expl = load_models()
    scaler  = prep["scaler"]
    le      = prep["label_encoder"]
    weights = prep.get("ensemble_weights", {"xgb":0.4,"lgbm":0.4,"rf":0.2})
    MODEL_READY = True
except Exception as e:
    MODEL_READY = False
    model_err   = str(e)


# ══════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════
def build_derived(row):
    mtk,fis,kim,bio = row[0],row[1],row[2],row[3]
    bind,bing       = row[4],row[5]
    eko,sos,sej     = row[6],row[7],row[8]
    seni            = row[9]
    riasec          = np.array(row[10:16], dtype=float)
    pem             = int(row[16])
    sains  = (mtk+fis+kim+bio)/4
    sosial = (eko+sos+sej)/3
    bahasa = (bind+bing)/2
    logika = 0.6*mtk + 0.4*fis
    sr     = np.sort(riasec)[::-1]
    dom_r  = int(np.argmax(riasec))
    gap    = float(sr[0]-sr[1])
    r_max  = float(sr[0]) if sr[0]>0 else 1
    return (list(row[:16]) +
            [sains, sosial, bahasa, seni, logika, pem, dom_r, gap,
             sains-sosial, sains-bahasa, bio-fis, kim-mtk, bing-bind,
             int(np.argmax([mtk,fis,kim,bio])), int(np.argmax([eko,sos,sej])),
             float(riasec[0]-riasec[1]), float(riasec[2]/r_max)])


def predict_rumpun(nilai_dict, peminatan, riasec_scores):
    pem_map = {"IPA": 0, "IPS": 1, "Bahasa/Seni": 2}
    vals = [float(nilai_dict.get(k, 0)) for k in MAPEL_ALL]
    r_arr = np.array([riasec_scores.get(c, 5.0) for c in "RIASEC"])
    raw   = vals + list(r_arr) + [pem_map.get(peminatan, 0)]
    feat  = build_derived(raw)
    Xs    = scaler.transform(np.array(feat, dtype=float).reshape(1, -1))

    p1 = xgb_model.predict_proba(Xs)[0]
    p2 = lgbm_model.predict_proba(Xs)[0]
    p3 = rf_model.predict_proba(Xs)[0]
    w  = weights
    prob = p1*w["xgb"] + p2*w["lgbm"] + p3*w["rf"]

    top5_idx = np.argsort(prob)[::-1][:5]
    results  = [{"rumpun": le.classes_[i], "prob": round(float(prob[i])*100, 1)}
                for i in top5_idx]

    # SHAP — handle semua format output dari berbagai versi shap/xgboost
    sv      = shap_expl.shap_values(Xs)
    n_class = len(le.classes_)
    top_cls = int(top5_idx[0])

    try:
        if isinstance(sv, list):
            # Format lama: list panjang n_class, tiap elemen (n_samples, n_feat)
            raw    = np.array(sv[top_cls])
            sv_top = raw[0] if raw.ndim == 2 else raw
        else:
            sv_arr = np.array(sv)
            if sv_arr.ndim == 3:
                # (n_class, n_samples, n_feat) atau (n_samples, n_feat, n_class)
                if sv_arr.shape[0] == n_class:
                    sv_top = sv_arr[top_cls, 0, :]
                elif sv_arr.shape[2] == n_class:
                    sv_top = sv_arr[0, :, top_cls]
                else:
                    sv_top = sv_arr[0, 0, :]
            elif sv_arr.ndim == 2:
                sv_top = sv_arr[0]
            else:
                sv_top = sv_arr.flatten()

        sv_top = np.array(sv_top, dtype=float).flatten()
        if len(sv_top) != len(FEATURE_NAMES):
            raise ValueError("shape mismatch")

    except Exception:
        sv_top = np.zeros(len(FEATURE_NAMES))

    shap_df = (
        pd.DataFrame({"fitur": FEATURE_NAMES, "kontribusi": sv_top})
        .sort_values("kontribusi", key=abs, ascending=False)
        .head(8)
    )
    return results, shap_df


def compute_riasec_scores(answers: dict) -> dict:
    """answers = {q_idx: True/False}.  Returns {R,I,A,S,E,C} in 0-10."""
    totals = {k: 0 for k in "RIASEC"}
    counts = {k: 0 for k in "RIASEC"}
    for idx, yes in answers.items():
        _, code = RIASEC_QUESTIONS[idx]
        counts[code] += 1
        if yes:
            totals[code] += 1
    scores = {}
    for k in "RIASEC":
        scores[k] = round((totals[k] / counts[k]) * 10, 2) if counts[k] > 0 else 5.0
    return scores


def top3_codes(scores: dict) -> str:
    return "".join(sorted(scores, key=scores.get, reverse=True)[:3])


def generate_riasec_analysis_stream(nama, code, scores, api_key):
    """Generator: stream token by token dari Google GenAI API."""
    if not GENAI_OK or not api_key:
        yield _fallback_analysis(nama, code, scores)
        return

    names = {k: RIASEC_FULL[k]["nama"] for k in "RIASEC"}
    skor  = {k: f"{scores[k]:.1f}/10" for k in "RIASEC"}
    top   = list(code)  # e.g. ['R','I','A']

    prompt = f"""Kamu adalah konselor pendidikan yang ramah dan memotivasi.
Tulis analisis kepribadian RIASEC untuk siswa bernama {nama}.
Kode Holland-nya: {code} ({' - '.join(names[c] for c in top)})

Skor RIASEC mereka:
{chr(10).join(f'  {k} ({names[k]}): {skor[k]}' for k in "RIASEC")}

Tulis dalam bahasa Indonesia yang hangat, personal, dan menyemangati.
Struktur tulisanmu HARUS persis seperti ini — gunakan heading dan paragraf:

## Hei, {nama}!
[2-3 kalimat pembuka yang menyambut {nama} dan memuji kombinasi {code}]

## {names[top[0]]} ({top[0]}) — {RIASEC_FULL[top[0]]["alias"]}
[2-3 paragraf mengalir tentang dimensi ini: kepribadian, cara belajar, lingkungan ideal]

## {names[top[1]]} ({top[1]}) — {RIASEC_FULL[top[1]]["alias"]}
[2-3 paragraf mengalir tentang dimensi ini: kepribadian, cara belajar, lingkungan ideal]

## {names[top[2]]} ({top[2]}) — {RIASEC_FULL[top[2]]["alias"]}
[2-3 paragraf mengalir tentang dimensi ini: kepribadian, cara belajar, lingkungan ideal]

## ✨ Pesan untuk {nama}
[1 paragraf motivasi yang menghubungkan ketiga dimensi]

ATURAN WAJIB:
- Jangan gunakan bullet list (* atau -)
- Tulis paragraf mengalir, bukan poin-poin
- Setiap heading HARUS diawali ## persis seperti di atas
- Jangan tambah heading lain selain yang sudah ditentukan"""

    MODELS_TO_TRY = ["gemini-2.5-flash"]
    client = genai.Client(api_key=api_key)
    last_err = None
    for model_id in MODELS_TO_TRY:
        try:
            for chunk in client.models.generate_content_stream(
                model=model_id,
                contents=prompt,
            ):
                if chunk.text:
                    yield chunk.text
            return  # sukses, selesai
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "quota" in err_str.lower():
                last_err = f"Kuota model {model_id} habis, mencoba model berikutnya..."
                continue  # coba model berikutnya
            else:
                yield f"__API_ERROR__:{e}"
                return
    # Semua model kena rate limit
    yield f"__API_ERROR__:Kuota Gemini API habis untuk hari ini (free tier limit 20 req/hari). Coba lagi besok atau upgrade ke paket berbayar di https://ai.dev"


def _fallback_analysis(nama, code, scores):
    """Fallback jika tidak ada API key."""
    top = list(code)
    lines = [f"**Kamu termasuk tipe {code}**\n"]
    lines.append(f"*{' - '.join(RIASEC_FULL[c]['nama'] for c in top)}*\n\n")
    lines.append(f"Hei {nama}! Berdasarkan jawaban kamu, profil kepribadianmu menunjukkan kombinasi yang unik.\n\n")
    for c in top:
        lines.append(f"**{RIASEC_FULL[c]['nama']} ({c}) — {RIASEC_FULL[c]['alias']}**\n\n")
        lines.append(RIASEC_FULL[c]['deskripsi'].replace("<b>","**").replace("</b>","**")
                     .replace("<br><br>","\n\n") + "\n\n")
    return "".join(lines)


def radar_rumpun(scores):
    cats = ["Realistic","Investigative","Artistic","Social","Enterprising","Conventional"]
    vals = [scores.get(k,0) for k in "RIASEC"]
    fig  = go.Figure(go.Scatterpolar(
        r=vals+[vals[0]], theta=cats+[cats[0]],
        fill="toself", name="Profil RIASEC",
        line=dict(color="#4f46e5",width=2.5),
        fillcolor="rgba(79,70,229,0.2)"))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0,10])),
        showlegend=False, height=350,
        paper_bgcolor="rgba(0,0,0,0)",
        title=dict(text="Radar RIASEC", x=0.5),
        margin=dict(t=50,b=20,l=30,r=30))
    return fig


# ══════════════════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════════════════
DEFAULTS = {
    "step":           1,
    "nama":           "",
    "peminatan":      "IPA",
    "nilai":          {k: 0 for k in MAPEL_ALL},
    "riasec_answers": {},     # {q_idx: True/False}
    "riasec_scores":  {},     # {R,I,A,S,E,C}
    "riasec_code":    "",
    "hasil_ml":       None,
    "shap_df":        None,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ══════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(
        '<div style="text-align:center;padding:1rem 0">'
        '<h2 style="margin:0;color:#4f46e5;font-weight:800">RekoJurusan AI</h2>'
        '<p style="color:#6b7280;font-size:0.82rem;margin:0.3rem 0 0">v2.0 — Rumpun Predictor</p>'
        '</div>', unsafe_allow_html=True)

    st.markdown("---")

    # Progress
    STEP_LABELS = ["Data & Nilai","Tes RIASEC (42Q)","Hasil Analisis"]
    for i, lbl in enumerate(STEP_LABELS):
        active = (i + 1) == st.session_state.step
        done   = (i + 1) < st.session_state.step
        bg  = "linear-gradient(90deg,#4f46e5,#7c3aed)" if active else ("#e0fdf4" if done else "#f1f5f9")
        clr = "white" if active else ("#059669" if done else "#64748b")
        fw  = "700" if active else "500"
        prefix = "▶ " if active else ("✓ " if done else "")
        st.markdown(
            f'<div style="padding:0.6rem 0.9rem;border-radius:10px;margin:0.25rem 0;'
            f'background:{bg};color:{clr};font-weight:{fw};font-size:0.88rem">'
            f'{prefix}{lbl}</div>', unsafe_allow_html=True)

    st.markdown("---")

    # API Key
    st.markdown("**Google AI API Key** (Gemini)")
    st.text_input(
        "API Key (untuk analisis LLM)",
        type="password",
        placeholder="AIza...",
        key="api_key",
        help="Opsional. Tanpa API key, analisis RIASEC menggunakan teks template.",
        label_visibility="collapsed"
    )
    if st.session_state.get("api_key", ""):
        st.success("API Key tersimpan sesi ini")
    else:
        st.info("Isi API key untuk analisis LLM personal")

    st.markdown("---")
    st.markdown(
        '<div style="font-size:0.76rem;color:#94a3b8;text-align:center">'
        '<b>Model:</b> XGBoost + LightGBM + RF<br>'
        '<b>Target:</b> 10 Rumpun Fakultas<br>'
        '<b>RIASEC:</b> 42 pertanyaan Ya/Tidak<br>'
        '<b>XAI:</b> SHAP TreeExplainer<br>'
        '<b>LLM:</b> Gemini LLM Model'
        '</div>', unsafe_allow_html=True)

    if not MODEL_READY:
        st.error("❌ Model gagal dimuat")
        with st.expander("Lihat detail error", expanded=True):
            st.code(model_err if "model_err" in dir() else "Unknown error", language="text")
            st.markdown("""
**Kemungkinan penyebab:**
- `gdown` belum terinstall → `pip install gdown`
- File Google Drive belum di-share publik (Anyone with the link)
- Kuota download Google Drive habis
- File `.pkl` corrupt atau tidak valid

**Solusi:** Pastikan semua file `.pkl` di Google Drive di-share dengan izin **Anyone with the link → Viewer**.
""")


# Baca api_key dari session state agar tersedia di semua step & tab
api_key = st.session_state.get("api_key", "")

# ══════════════════════════════════════════════════════════════
#  HERO
# ══════════════════════════════════════════════════════════════
st.markdown(
    '<div class="hero">'
    '<h1>RekoJurusan AI</h1>'
    '<p>Temukan rumpun & jurusan terbaikmu — RIASEC 42Q · AI Ensemble · LLM Analysis</p>'
    '</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
#  STEP 1 — DATA DIRI & NILAI
# ══════════════════════════════════════════════════════════════
if st.session_state.step == 1:
    st.markdown('<div class="step-badge">Langkah 1 dari 3 — Data Diri & Nilai Akademik</div>',
                unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1.3])

    with c1:
        st.markdown("#### Data Diri")
        nama = st.text_input("Nama Lengkap", value=st.session_state.nama,
                             placeholder="Nama kamu ...")
        pem  = st.selectbox("Peminatan / Jurusan SMA",
                             ["IPA", "IPS", "Bahasa/Seni"],
                             index=["IPA","IPS","Bahasa/Seni"].index(st.session_state.peminatan))
        st.info("Masukkan nilai rapor rata-rata (skala 0–100).")

    with c2:
        st.markdown("#### Nilai Mata Pelajaran")
        mpl_show = {
            "IPA":        ["matematika","fisika","kimia","biologi","bahasa_indonesia","bahasa_inggris"],
            "IPS":        ["matematika","ekonomi","sosiologi","sejarah","bahasa_indonesia","bahasa_inggris"],
            "Bahasa/Seni":["bahasa_indonesia","bahasa_inggris","seni_budaya","matematika","sosiologi","sejarah"],
        }[pem]

        nilai_dict = dict(st.session_state.nilai)
        cc = st.columns(2)
        for i, mp in enumerate(mpl_show):
            with cc[i % 2]:
                val = st.number_input(
                    MAPEL_LABEL[mp], 0, 100,
                    int(st.session_state.nilai.get(mp, 0)), 1, key=f"n_{mp}")
                nilai_dict[mp] = val
                if val == 0:
                    st.markdown(
                        f'<div style="color:#ef4444;font-size:0.75rem;margin-top:-0.5rem;margin-bottom:0.3rem">'
                        f'&#9888; Nilai {MAPEL_LABEL[mp]} tidak boleh 0</div>',
                        unsafe_allow_html=True)
        for mp in MAPEL_ALL:
            if mp not in nilai_dict:
                nilai_dict[mp] = st.session_state.nilai.get(mp, 0)

    # Metrics — hitung hanya dari mapel yang ditampilkan sesuai peminatan
    st.markdown("---")
    vals  = [nilai_dict[mp] for mp in mpl_show]
    avg_v = sum(vals) / len(vals)
    mc1,mc2,mc3,mc4 = st.columns(4)
    for col, (lbl, v) in zip([mc1,mc2,mc3,mc4],
        [("Rata-rata",f"{avg_v:.1f}"),("Tertinggi",str(max(vals))),
         ("Terendah",str(min(vals))),
         ("Grade","A" if avg_v>=85 else "B" if avg_v>=75 else "C" if avg_v>=65 else "D")]):
        with col:
            st.markdown(f'<div class="metric-box"><div class="val">{v}</div>'
                        f'<div class="lbl">{lbl}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    _, bc, _ = st.columns([3,1,3])
    with bc:
        if st.button("Lanjut ke Tes RIASEC \u2192", use_container_width=True, type="primary"):
            if not nama.strip():
                st.error("\u26a0\ufe0f Nama tidak boleh kosong!")
            else:
                nilai_kosong = [MAPEL_LABEL[mp] for mp in mpl_show if nilai_dict.get(mp, 0) == 0]
                if nilai_kosong:
                    st.error(
                        f"\u26a0\ufe0f Nilai tidak boleh 0 untuk: **{', '.join(nilai_kosong)}**. "
                        f"Masukkan nilai rapor yang sebenarnya (1\u2013100).")
                else:
                    st.session_state.nama     = nama
                    st.session_state.peminatan= pem
                    st.session_state.nilai    = nilai_dict
                    st.session_state.step     = 2
                    st.rerun()


# ══════════════════════════════════════════════════════════════
#  STEP 2 — 42 PERTANYAAN RIASEC
# ══════════════════════════════════════════════════════════════
elif st.session_state.step == 2:
    st.markdown('<div class="step-badge">Langkah 2 dari 3 — Tes Minat RIASEC (42 Pertanyaan)</div>',
                unsafe_allow_html=True)

    st.markdown(
        '<div style="background:#f0f9ff;border:1.5px solid #bae6fd;border-radius:12px;'
        'padding:1rem 1.2rem;margin-bottom:1.5rem;color:#0f172a;">'
        '<b>Petunjuk:</b> Jawab setiap pernyataan dengan jujur. '
        'Centang <b>YA</b> jika pernyataan sesuai dengan dirimu, biarkan kosong jika <b>TIDAK</b>. '
        'Tidak ada jawaban benar atau salah!'
        '</div>', unsafe_allow_html=True)

    # Tampilkan live score di atas
    answers = dict(st.session_state.riasec_answers)

    # Hitung skor realtime dari jawaban yang sudah ada
    live_scores = compute_riasec_scores(answers) if answers else {k:5.0 for k in "RIASEC"}

    # Progress
    # Hitung langsung dari widget session state (akurat saat uncheck)
    n_yes = sum(1 for qi in range(42) if st.session_state.get(f"q_{qi}", False) is True)
    n_tidak = 42 - n_yes
    st.progress(n_yes / 42, text=f"Dicentang (YA): {n_yes} / 42  ·  Belum dicentang: {n_tidak}")

    # Group questions by section (6 sections × 7Q)
    SECTION_LABELS = [
        ("Realistic — Praktis & Fisik",         "R",  0,  7),
        ("Investigative — Analitis & Ilmiah",   "I",  7, 14),
        ("Artistic — Kreatif & Ekspresif",       "A", 14, 21),
        ("Social — Sosial & Peduli",             "S", 21, 28),
        ("Enterprising — Kepemimpinan & Bisnis", "E", 28, 35),
        ("Conventional — Terstruktur & Detail",  "C", 35, 42),
    ]

    # 2 kolom, masing-masing 3 section
    left_secs  = SECTION_LABELS[:3]
    right_secs = SECTION_LABELS[3:]
    col_l, col_r = st.columns(2)

    for col, sections in [(col_l, left_secs), (col_r, right_secs)]:
        with col:
            for sec_label, code, q_start, q_end in sections:
                clr = RIASEC_FULL[code]["warna"]
                st.markdown(
                    f'<div style="font-weight:700;color:{clr};font-size:0.95rem;'
                    f'border-left:4px solid {clr};padding:0.4rem 0.8rem;'
                    f'background:{clr}15;border-radius:0 8px 8px 0;margin:1rem 0 0.5rem">'
                    f'{sec_label}</div>', unsafe_allow_html=True)
                for qi in range(q_start, q_end):
                    q_text, _ = RIASEC_QUESTIONS[qi]
                    checked = answers.get(qi, False)
                    new_val = st.checkbox(
                        q_text, value=checked, key=f"q_{qi}")
                    answers[qi] = new_val

    # Update scores in realtime
    st.session_state.riasec_answers = answers
    live_scores = compute_riasec_scores(answers)

    # Live RIASEC bar preview
    st.markdown("---")
    st.markdown("#### Profil RIASEC Kamu Saat Ini")
    bar_cols = st.columns(6)
    for i, code in enumerate("RIASEC"):
        sc    = live_scores.get(code, 0)
        clr   = RIASEC_FULL[code]["warna"]
        with bar_cols[i]:
            st.markdown(
                f'<div style="text-align:center">'
                f'<div style="font-weight:800;color:{clr};font-size:1.2rem">{code}</div>'
                f'<div style="font-size:0.75rem;color:#64748b">{RIASEC_FULL[code]["nama"]}</div>'
                f'<div style="font-size:1.4rem;font-weight:800;color:{clr};margin-top:0.3rem">'
                f'{sc:.1f}</div><div style="font-size:0.7rem;color:#94a3b8">/10</div>'
                f'<div style="background:#e2e8f0;border-radius:999px;height:8px;margin-top:0.4rem;overflow:hidden">'
                f'<div style="width:{sc*10:.0f}%;background:{clr};height:100%;border-radius:999px"></div>'
                f'</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    _, b1, b2, _ = st.columns([2,1,1,2])
    with b1:
        if st.button("← Kembali", use_container_width=True):
            st.session_state.step = 1; st.rerun()
    with b2:
        if st.button("Lihat Hasil \u2192", use_container_width=True, type="primary"):
            n_yes_submit = sum(1 for qi in range(42) if st.session_state.get(f"q_{qi}", False) is True)
            if n_yes_submit == 0:
                st.warning(
                    "\u26a0\ufe0f Kamu belum mencentang satu pun pertanyaan. "
                    "Centang 'YA' untuk pernyataan yang sesuai dengan dirimu sebelum melanjutkan.")
            else:
                st.session_state.riasec_scores = live_scores
                st.session_state.riasec_code   = top3_codes(live_scores)
                # Run ML prediction
                if MODEL_READY:
                    with st.spinner("AI menganalisis profil kamu ..."):
                        hasil_ml, shap_df = predict_rumpun(
                            st.session_state.nilai,
                            st.session_state.peminatan,
                            live_scores)
                    st.session_state.hasil_ml = hasil_ml
                    st.session_state.shap_df  = shap_df.to_dict()
                st.session_state.step = 3
                st.rerun()


# ══════════════════════════════════════════════════════════════
#  STEP 3 — HASIL ANALISIS
# ══════════════════════════════════════════════════════════════
elif st.session_state.step == 3:
    nama         = st.session_state.nama
    code         = st.session_state.riasec_code or "RIA"
    scores       = st.session_state.riasec_scores
    hasil_ml     = st.session_state.hasil_ml or []
    shap_df_dict = st.session_state.shap_df or {}

    # ── Header profil ────────────────────────────────────────
    top_codes  = list(code)
    top_names  = " · ".join(f"{RIASEC_FULL[c]['nama']}" for c in top_codes)
    st.markdown(
        f'<div style="background:linear-gradient(135deg,#0f172a,#1e1b4b);color:white;'
        f'padding:1.8rem;border-radius:20px;margin-bottom:1.5rem;text-align:center">'
        f'<div style="font-size:0.85rem;opacity:0.6;margin-bottom:0.3rem">Hasil Analisis untuk</div>'
        f'<div style="font-size:2rem;font-weight:900">{nama}</div>'
        f'<div style="margin-top:0.6rem;font-size:0.9rem;opacity:0.8">{top_names}</div>'
        f'<div style="display:flex;justify-content:center;gap:0.8rem;margin-top:0.8rem;flex-wrap:wrap">'
        f'<span style="background:rgba(79,70,229,0.3);padding:0.3rem 0.9rem;'
        f'border-radius:8px;font-size:0.82rem">{st.session_state.peminatan}</span>'
        f'<span style="background:rgba(245,158,11,0.3);padding:0.3rem 0.9rem;'
        f'border-radius:8px;font-size:0.82rem">Holland Code: {code}</span>'
        f'</div></div>', unsafe_allow_html=True)

    # ── TABS ─────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(
        ["Analisis Kepribadian (RIASEC)", "Rekomendasi Rumpun (ML)", "Summary & Chatbot"])

    # ── TAB 1: RIASEC LLM Analysis ───────────────────────────
    with tab1:
        st.markdown("#### Analisis Kepribadian Personalmu")

        _tab1_api_key = st.session_state.get("api_key", "")

        def _show_fallback_ui(nama, code, top_codes):
            """Tampilkan analisis statis per dimensi dengan UI yang proper."""
            st.markdown(
                f'<div class="chat-bubble">'
                f'<div style="font-size:0.85rem;opacity:0.7;margin-bottom:0.8rem">Tipe Holland kamu:</div>'
                f'<span class="type-code">{code}</span>'
                f'<div class="type-name">{" – ".join(RIASEC_FULL[c]["nama"] for c in top_codes)}</div>'
                f'<div style="font-size:0.9rem;opacity:0.85;margin-bottom:0.5rem">'
                f'Hei <b>{nama}</b>! Profil kepribadianmu menunjukkan kombinasi yang menarik dan unik. '
                f'Berikut penjelasan lengkap tentang tipe Holland kamu:</div>'
                f'</div>', unsafe_allow_html=True)
            for c in top_codes:
                info = RIASEC_FULL[c]
                st.markdown(
                    f'<div class="personality-box" style="border-color:{info["warna"]}">'
                    f'<div class="box-title" style="color:{info["warna"]}">'
                    f'{info["nama"]} ({c}) — {info["alias"]}</div>'
                    f'<div class="box-body">'
                    f'{info["deskripsi"]}</div></div>', unsafe_allow_html=True)

        if GENAI_OK and _tab1_api_key:
            # Badge info
            st.markdown(
                '<div style="background:rgba(79,70,229,0.07);border:1.5px dashed #818cf8;'
                'border-radius:10px;padding:0.8rem 1.2rem;margin-bottom:1rem;font-size:0.85rem;color:#4f46e5">'
                '<b>Analisis personal</b> dihasilkan oleh <b>Gemini</b> berdasarkan profil RIASEC kamu'
                '</div>', unsafe_allow_html=True)

            full_text = ""
            api_error_msg = None

            stream_placeholder = st.empty()

            with st.spinner("Gemini sedang menganalisis kepribadianmu..."):
                for chunk in generate_riasec_analysis_stream(nama, code, scores, _tab1_api_key):
                    if chunk.startswith("__API_ERROR__:"):
                        api_error_msg = chunk[len("__API_ERROR__:"):]
                        break
                    full_text += chunk
                    # Saat streaming: tampilkan teks mentah dulu agar user melihat progress
                    stream_placeholder.markdown(full_text)

            def _render_llm_sections(raw_text, top_codes_list):
                """Split output Gemini pada ## heading lalu render tiap bagian di box terpisah."""
                import re

                # Split pada setiap ## heading — hasilkan list [("## Judul", "isi..."), ...]
                parts = re.split(r'(?m)^(##[^\n]+)', raw_text)
                # parts[0] = teks sebelum heading pertama (biasanya kosong)
                # parts[1], parts[2] = heading, isi; parts[3], parts[4] = heading, isi; dst

                sections = []  # list of (heading_text, body_text)
                i = 1
                while i < len(parts) - 1:
                    heading = parts[i].strip().lstrip("#").strip()
                    body    = parts[i+1].strip()
                    sections.append((heading, body))
                    i += 2

                # Jika tidak ada ## heading sama sekali → fallback 1 box
                if not sections:
                    st.markdown(
                        f'<div style="background:#1e1b4b;color:white;border-radius:14px;'
                        f'padding:1.5rem 1.8rem;margin:1rem 0;line-height:1.7">'
                        f'{raw_text}'
                        f'</div>', unsafe_allow_html=True)
                    return

                for idx, (heading, body) in enumerate(sections):
                    # Deteksi apakah heading ini milik dimensi RIASEC tertentu
                    matched_code = None
                    for c in top_codes_list:
                        info = RIASEC_FULL[c]
                        if f"({c})" in heading or info["nama"].lower() in heading.lower():
                            matched_code = c
                            break

                    is_intro    = idx == 0
                    is_motivasi = "pesan" in heading.lower() or "✨" in heading or idx == len(sections)-1 and matched_code is None

                    if matched_code:
                        # ── Box per dimensi RIASEC ────────────────────────────
                        info     = RIASEC_FULL[matched_code]
                        warna    = info["warna"]
                        skor_val = scores.get(matched_code, 0)
                        skor_pct = f'{skor_val*10:.0f}%'
                        with st.container():
                            st.markdown(
                                f'<div style="background:#1e293b;border-left:5px solid {warna};'
                                f'border-radius:12px;padding:1.3rem 1.6rem;margin:0.7rem 0">'
                                f'<div style="display:flex;align-items:center;gap:0.7rem;margin-bottom:0.8rem">'
                                f'<span style="font-size:1.3rem">{info["emoji"]}</span>'
                                f'<div>'
                                f'<div style="font-weight:800;font-size:0.95rem;color:{warna}">{heading}</div>'
                                f'<div style="font-size:0.75rem;color:#94a3b8;margin-top:0.1rem">'
                                f'Skor: {skor_val:.1f}/10 '
                                f'<span style="display:inline-block;margin-left:0.4rem;background:#0f172a;'
                                f'border-radius:999px;height:6px;width:60px;vertical-align:middle;overflow:hidden">'
                                f'<span style="display:block;width:{skor_pct};background:{warna};'
                                f'height:100%;border-radius:999px"></span></span>'
                                f'</div></div></div></div>',
                                unsafe_allow_html=True)
                            st.markdown(
                                f'<div style="background:#1e293b;border-left:5px solid {warna};'
                                f'border-radius:0 0 12px 12px;padding:0 1.6rem 1.3rem;'
                                f'margin:-0.75rem 0 0.7rem;color:#cbd5e1;font-size:0.9rem;line-height:1.8">'
                                f'{body}</div>',
                                unsafe_allow_html=True)

                    elif is_motivasi:
                        # ── Box motivasi penutup ──────────────────────────────
                        st.markdown(
                            f'<div style="background:linear-gradient(135deg,#0f2027,#203a43,#2c5364);'
                            f'border-radius:12px;padding:1.2rem 1.6rem;margin:0.7rem 0;'
                            f'border:1px solid rgba(16,185,129,0.3)">'
                            f'<div style="font-weight:700;color:#34d399;margin-bottom:0.5rem">'
                            f'✨ {heading}</div>'
                            f'<div style="color:#d1fae5;font-size:0.9rem;line-height:1.8">{body}</div>'
                            f'</div>', unsafe_allow_html=True)

                    else:
                        # ── Box intro / header ────────────────────────────────
                        st.markdown(
                            f'<div style="background:#1e1b4b;color:white;border-radius:14px;'
                            f'padding:1.5rem 1.8rem;margin:0 0 1rem 0;line-height:1.75">'
                            f'<div style="font-size:0.8rem;opacity:0.55;margin-bottom:0.4rem">Tipe Holland kamu</div>'
                            f'<div style="font-size:2rem;font-weight:900;letter-spacing:0.1em;'
                            f'background:linear-gradient(90deg,#f59e0b,#ec4899,#8b5cf6);'
                            f'-webkit-background-clip:text;-webkit-text-fill-color:transparent;display:inline-block">'
                            f'{code}</div>'
                            f'<div style="font-size:0.9rem;color:#c7d2fe;margin:0.3rem 0 0.8rem">'
                            f'{" · ".join(RIASEC_FULL[c]["nama"] for c in top_codes_list)}</div>'
                            f'<div style="font-size:0.92rem;opacity:0.88;line-height:1.75">{body}</div>'
                            f'</div>', unsafe_allow_html=True)

            if api_error_msg:
                stream_placeholder.empty()
                st.error(
                    f"⚠️ Gagal mengambil analisis dari Gemini AI. "
                    f"Kemungkinan API key tidak valid atau kuota habis.\n\n"
                    f"**Detail error:** `{api_error_msg}`"
                )
                st.info("Menampilkan analisis statis berdasarkan tipe RIASEC kamu:")
                _show_fallback_ui(nama, code, top_codes)
            elif full_text:
                stream_placeholder.empty()
                _render_llm_sections(full_text, top_codes)
        else:
            # Tidak ada API key — tampilkan fallback langsung
            _show_fallback_ui(nama, code, top_codes)
            if not _tab1_api_key:
                st.info("Isi **Google AI API Key (Gemini)** di sidebar untuk mendapatkan analisis personal yang lebih mendalam dari Gemini AI.")

        # RIASEC Radar
        st.markdown("---")
        col_r1, col_r2 = st.columns([1, 1])
        with col_r1:
            st.plotly_chart(radar_rumpun(scores), use_container_width=True)
        with col_r2:
            st.markdown("#### Skor Detail RIASEC")
            for code_k in sorted(scores, key=scores.get, reverse=True):
                s    = scores[code_k]
                info = RIASEC_FULL[code_k]
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:0.8rem;margin:0.4rem 0">'
                    f'<span style="width:1.5rem;font-weight:800;color:{info["warna"]}">{code_k}</span>'
                    f'<div style="flex:1;background:#e2e8f0;border-radius:999px;height:10px;overflow:hidden">'
                    f'<div style="width:{s*10:.0f}%;background:{info["warna"]};height:100%;border-radius:999px"></div>'
                    f'</div>'
                    f'<span style="width:3.5rem;text-align:right;font-weight:700;color:{info["warna"]}">{s:.1f}/10</span>'
                    f'</div>', unsafe_allow_html=True)

    # ── TAB 2: ML Rumpun Recommendation ──────────────────────
    with tab2:
        if not MODEL_READY:
            st.error(f" Model belum tersedia. Jalankan `train_rumpun.py` terlebih dahulu.")
        elif not hasil_ml:
            st.warning(" Hasil ML tidak tersedia.")
        else:
            st.markdown("#### Top 5 Rumpun Terbaik Untukmu")

            rank_cls = ["rank-1","rank-2","rank-3","rank-1","rank-2"]
            medals   = ["🥇","🥈","🥉","4️⃣","5️⃣"]

            for i, r in enumerate(hasil_ml[:5]):
                rump = r["rumpun"]
                info = RUMPUN_INFO.get(rump, {})
                dom_r_str = " · ".join(
                    f'<span style="background:{RIASEC_FULL[k]["warna"]}20;'
                    f'color:{RIASEC_FULL[k]["warna"]};padding:2px 8px;border-radius:999px;'
                    f'font-weight:700;font-size:0.75rem">{k}</span>'
                    for k in info.get("riasec", []))

                jurusan_pills = "".join(
                    f'<span style="display:inline-block;background:#334155;border:1px solid #475569;'
                    f'border-radius:6px;padding:0.2rem 0.6rem;margin:0.2rem;'
                    f'font-size:0.76rem;font-weight:600;color:#cbd5e1">{j}</span>'
                    for j in info.get("jurusan", [])[:6])

                st.markdown(
                    f'<div class="rumpun-card {rank_cls[i % 3]}">'
                    f'<div style="display:flex;justify-content:space-between;align-items:start;gap:1rem">'
                    f'<div style="flex:1">'
                    f'<div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:0.4rem">'
                    f'<div>'
                    f'<div style="font-weight:800;font-size:1.1rem;color:#f1f5f9">{rump}</div>'
                    f'<div style="font-size:0.82rem;color:#94a3b8">{info.get("deskripsi","")}</div>'
                    f'</div></div>'
                    f'<div style="margin:0.5rem 0 0.3rem">{dom_r_str}</div>'
                    f'<div style="font-size:0.82rem;color:#94a3b8;margin:0.3rem 0">'
                    f'Karir: {info.get("karir","")}</div>'
                    f'<div style="margin-top:0.4rem">{jurusan_pills}</div>'
                    f'</div>'
                    f'<div style="text-align:right;min-width:80px">'
                    f'<div style="font-size:2rem;font-weight:900;color:{info.get("warna","#4f46e5")}">'
                    f'{r["prob"]}%</div>'
                    f'<div style="font-size:0.75rem;color:#64748b">kecocokan</div>'
                    f'</div></div></div>', unsafe_allow_html=True)

                st.progress(r["prob"]/100)
                st.markdown("")

            # Nilai Academic radar
            st.markdown("---")
            st.markdown("#### Profil Nilai Akademik")
            vals_n = [st.session_state.nilai.get(k, 0) for k in MAPEL_ALL]
            fig_n  = go.Figure(go.Scatterpolar(
                r=vals_n+[vals_n[0]],
                theta=NILAI_LABELS+[NILAI_LABELS[0]],
                fill="toself", name="Nilaimu",
                line=dict(color="#f59e0b",width=2.5),
                fillcolor="rgba(245,158,11,0.2)"))
            fig_n.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0,100])),
                showlegend=False, height=380,
                paper_bgcolor="rgba(0,0,0,0)",
                title=dict(text="Radar Nilai Akademik", x=0.5))
            st.plotly_chart(fig_n, use_container_width=True)

    # ── TAB 3: SUMMARY & CHATBOT ─────────────────────────────
    with tab3:

        # ── SUMMARY ──────────────────────────────────────────
        st.markdown("#### 📋 Ringkasan Profil Kamu")

        col_sum1, col_sum2 = st.columns(2)
        with col_sum1:
            st.markdown("**Nilai Akademik:**")
            for k in MAPEL_ALL:
                v = st.session_state.nilai.get(k, 0)
                bar_w = int(v)
                color = "#4ade80" if v>=80 else "#fbbf24" if v>=65 else "#f87171"
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:0.5rem;margin:0.2rem 0">'
                    f'<span style="width:100px;font-size:0.8rem">{MAPEL_LABEL[k]}</span>'
                    f'<div style="flex:1;background:#e2e8f0;border-radius:999px;height:8px;overflow:hidden">'
                    f'<div style="width:{bar_w}%;background:{color};height:100%;border-radius:999px"></div>'
                    f'</div><span style="width:35px;text-align:right;font-size:0.8rem;font-weight:700">{v}</span>'
                    f'</div>', unsafe_allow_html=True)

        with col_sum2:
            st.markdown("**Skor RIASEC:**")
            for code_k in "RIASEC":
                s    = scores.get(code_k, 0)
                info = RIASEC_FULL[code_k]
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:0.5rem;margin:0.25rem 0">'
                    f'<span style="width:22px;font-weight:800;color:{info["warna"]}">{code_k}</span>'
                    f'<span style="width:80px;font-size:0.8rem;color:#64748b">{info["nama"]}</span>'
                    f'<div style="flex:1;background:#e2e8f0;border-radius:999px;height:8px;overflow:hidden">'
                    f'<div style="width:{s*10:.0f}%;background:{info["warna"]};height:100%;border-radius:999px"></div>'
                    f'</div><span style="width:40px;text-align:right;font-size:0.8rem;font-weight:700;'
                    f'color:{info["warna"]}">{s:.1f}</span>'
                    f'</div>', unsafe_allow_html=True)

            # Top rumpun summary
            if hasil_ml:
                st.markdown("<br>**Top 3 Rumpun Rekomendasi:**", unsafe_allow_html=True)
                for i, r in enumerate(hasil_ml[:3]):
                    medals = ["🥇","🥈","🥉"]
                    rump_info = RUMPUN_INFO.get(r["rumpun"], {})
                    st.markdown(
                        f'<div style="display:flex;align-items:center;gap:0.6rem;'
                        f'background:#f8fafc;border-radius:8px;padding:0.4rem 0.7rem;margin:0.25rem 0;'
                        f'border-left:3px solid {rump_info.get("warna","#4f46e5")}">'
                        f'<span>{medals[i]}</span>'
                        f'<span style="flex:1;font-size:0.85rem;font-weight:600;color:#0f172a">{r["rumpun"]}</span>'
                        f'<span style="font-weight:800;color:{rump_info.get("warna","#4f46e5")}">'
                        f'{r["prob"]}%</span></div>', unsafe_allow_html=True)

        # ── CHATBOT ──────────────────────────────────────────
        st.markdown("---")
        st.markdown(
            '<div style="background:linear-gradient(135deg,#0f172a,#1e1b4b);color:white;'
            'padding:1.2rem 1.5rem;border-radius:14px;margin-bottom:1rem">'
            '<div style="font-size:1.1rem;font-weight:800">🤖 Konselor AI RekoJurusan</div>'
            '<div style="font-size:0.82rem;opacity:0.7;margin-top:0.3rem">'
            'Tanya apapun tentang prodi yang cocok, peluang karir, atau lintas jurusan '
            'berdasarkan profilmu</div></div>',
            unsafe_allow_html=True)

        _api_key = st.session_state.get("api_key", "")
        if not GENAI_OK or not _api_key:
            st.warning(
                "Isi **Google AI API Key (Gemini)** di sidebar untuk mengaktifkan Chatbot AI. "
                "Tanpa API key, fitur chatbot tidak tersedia.")
        else:
            # ── Init chat history ─────────────────────────────
            if "chat_messages" not in st.session_state:
                st.session_state.chat_messages = []

            # Build system context from user profile
            nilai_summary = ", ".join(
                f"{MAPEL_LABEL[k]}={st.session_state.nilai.get(k,0)}"
                for k in MAPEL_ALL
                if st.session_state.nilai.get(k, 0) > 0
            )
            if not nilai_summary:
                nilai_summary = "(belum diisi)"
            riasec_summary = ", ".join(
                f"{k}={scores.get(k,0):.1f}/10" for k in "RIASEC")
            top3_rumpun_str = "; ".join(
                f"{r['rumpun']} ({r['prob']}%)"
                for r in (hasil_ml[:3] if hasil_ml else []))
            all_jurusan_str = "; ".join(
                f"{rump}: {', '.join(info['jurusan'][:5])}"
                for rump, info in RUMPUN_INFO.items())

            SYSTEM_PROMPT = f"""Kamu adalah konselor pendidikan AI yang ramah, empatik, dan membantu siswa Indonesia memilih jurusan kuliah.

PROFIL SISWA:
- Nama: {st.session_state.nama}
- Peminatan SMA: {st.session_state.peminatan}
- Catatan nilai: Hanya mata pelajaran yang diambil siswa yang tercantum.Nilai yang tidak muncul berarti mata pelajaran tersebut TIDAK diambil karena beda jurusan/peminatan, BUKAN berarti nilainya buruk atau nol. Jangan pernah menyebut nilai 0 atau mengasumsikan siswa lemah di bidang yang tidak ada datanya.
- Nilai Akademik: {nilai_summary}
- Skor RIASEC: {riasec_summary}
- Kode Holland Dominan: {st.session_state.riasec_code}
- Top 3 Rumpun Rekomendasi AI: {top3_rumpun_str}

DATA PRODI & RUMPUN:
{all_jurusan_str}

TUGASMU:
1. Bantu siswa memahami prodi-prodi yang cocok berdasarkan profil mereka — berikan contoh konkret prodi, kampus (sebutkan PTN/PTS ternama di Indonesia), dan gambaran kehidupan kuliah.
2. Jelaskan prospek karir untuk setiap prodi yang ditanyakan — gaji rata-rata, peluang kerja, bidang industri.
3. Fasilitasi pertanyaan lintas jurusan — jika siswa ingin pindah jalur (misal dari IPA ke Komunikasi, atau dari teknik ke bisnis), analisis apakah nilai dan RIASEC mendukung, serta langkah-langkah yang perlu disiapkan (mata pelajaran penunjang, portofolio, dsb).
4. Berikan saran yang jujur, realistis, dan menyemangati.

GAYA KOMUNIKASI:
- Gunakan bahasa Indonesia yang hangat, santai namun informatif
- Panggil siswa dengan nama mereka ({st.session_state.nama})
- Berikan contoh spesifik, bukan jawaban generik
- Jika ditanya tentang lintas jurusan, selalu dasarkan pada analisis nilai dan RIASEC yang sudah tersedia
- Respons singkat dan padat (maksimal 300 kata per pesan), kecuali diminta lebih detail

FORMAT OUTPUT — WAJIB DIIKUTI:
- Tulis dalam paragraf mengalir, BUKAN bullet list dengan tanda * atau -
- Jika ingin menyebut beberapa item (kampus, prodi, karir), tulis dalam kalimat: "seperti UI, UGM, dan ITS" — bukan daftar berpoin
- Gunakan **teks tebal** hanya untuk nama prodi/rumpun/kampus yang penting
- Pisahkan topik dengan baris kosong (enter dua kali) supaya mudah dibaca
- JANGAN gunakan * sebagai bullet di awal baris sama sekali"""

            # ── Helper: panggil Gemini streaming, return generator ──
            def _stream_reply(history, last_msg):
                """Yield token dari Gemini stream."""
                gemini_history = []
                for m in history:
                    role = "model" if m["role"] == "assistant" else "user"
                    gemini_history.append({"role": role, "parts": [{"text": m["content"]}]})

                full_prompt = SYSTEM_PROMPT + "\n\n---\n\n" + last_msg
                if gemini_history:
                    contents = gemini_history + [{"role": "user", "parts": [{"text": full_prompt}]}]
                else:
                    contents = full_prompt

                client = genai.Client(api_key=_api_key)
                MODELS = ["gemini-2.5-flash"]
                for model_id in MODELS:
                    try:
                        for chunk in client.models.generate_content_stream(
                            model=model_id,
                            contents=contents,
                        ):
                            if chunk.text:
                                yield chunk.text
                        return  # sukses
                    except Exception as e:
                        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e) or "quota" in str(e).lower():
                            continue
                        raise  # error lain, lempar ke caller
                raise Exception("Kuota Gemini API habis untuk hari ini (free tier limit). Coba lagi besok atau upgrade akun di https://ai.dev")

            def _fmt(text):
                """Bersihkan output Gemini: ubah bullet * jadi •, pastikan newline ganda."""
                import re
                # Ganti baris yang dimulai "* " menjadi "• " agar tidak crash Markdown
                text = re.sub(r'(?m)^\* ', '\u2022 ', text)
                # Single newline → double newline supaya Markdown render sebagai paragraf baru
                nl = '\n'
                text = re.sub(r'(?<!' + nl + r')' + nl + r'(?!' + nl + r')', nl + nl, text)
                return text.strip()

            # ── Tampilkan riwayat chat (pakai st.chat_message native) ──
            for msg in st.session_state.chat_messages:
                with st.chat_message("user" if msg["role"] == "user" else "assistant",
                                     avatar="🧑" if msg["role"] == "user" else "🤖"):
                    txt = _fmt(msg["content"]) if msg["role"] == "assistant" else msg["content"]
                    st.markdown(txt)

            # ── Input chat utama (di bawah riwayat) ────────────────────
            user_input = st.chat_input(
                "Tanya tentang prodi, karir, atau lintas jurusan...",
                key="chatbot_input")
            if user_input and user_input.strip():
                user_text = user_input.strip()
                st.session_state.chat_messages.append({"role": "user", "content": user_text})
                with st.chat_message("user", avatar="🧑"):
                    st.markdown(user_text)
                with st.chat_message("assistant", avatar="🤖"):
                    try:
                        reply = st.write_stream(
                            _stream_reply(
                                st.session_state.chat_messages[:-1], user_text))
                        st.session_state.chat_messages.append(
                            {"role": "assistant", "content": _fmt(str(reply))})
                    except Exception as e:
                        st.error(f"Gagal menghubungi Gemini AI: {e}")

 #          # ── Suggested questions (hanya saat chat kosong) ───────────
            if not st.session_state.chat_messages:
                st.markdown(
                    '<div style="font-size:0.83rem;color:#64748b;margin:0.5rem 0 0.4rem">'
                    '💡 <b>Pertanyaan populer — klik untuk mulai:</b></div>',
                    unsafe_allow_html=True)
                suggestions = [
                    "Prodi apa yang paling cocok untuk saya berdasarkan nilai dan RIASEC saya?",
                    f"Bagaimana prospek karir di {hasil_ml[0]['rumpun'] if hasil_ml else 'Teknik & Informatika'}?",
                    "Saya ingin lintas jurusan, apakah nilai saya mendukung?",
                    "Kampus mana yang bagus untuk jurusan rekomendasi saya?",
                ]
                sug_cols = st.columns(2)
                for i, sug in enumerate(suggestions):
                    with sug_cols[i % 2]:
                        if st.button(sug, key=f"sug_{i}", use_container_width=True):
                            st.session_state["pending_sug"] = sug
                            st.rerun()

            # ── Proses pending suggestion (setelah rerun, tombol sudah hilang) ──
            if st.session_state.get("pending_sug"):
                sug = st.session_state.pop("pending_sug")
                st.session_state.chat_messages.append({"role": "user", "content": sug})
                with st.chat_message("user", avatar="🧑"):
                    st.markdown(sug)
                with st.chat_message("assistant", avatar="🤖"):
                    try:
                        reply = st.write_stream(
                            _stream_reply(
                                st.session_state.chat_messages[:-1], sug))
                        st.session_state.chat_messages.append(
                            {"role": "assistant", "content": _fmt(str(reply))})
                    except Exception as e:
                        st.error(f"Gagal menghubungi Gemini AI: {e}")

            # ── Tombol hapus riwayat ────────────────────────────────────
            if st.session_state.chat_messages:
                if st.button("🗑️ Hapus Riwayat Chat", key="clear_chat"):
                    st.session_state.chat_messages = []
                    st.rerun()

    # ── Bottom actions ────────────────────────────────────────
    st.markdown("---")
    _, bc1, bc2, _ = st.columns([2,1,1,2])
    with bc1:
        if st.button("← Ubah Jawaban", use_container_width=True):
            st.session_state.step = 2; st.rerun()
    with bc2:
        if st.button("Mulai Ulang", use_container_width=True):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()
