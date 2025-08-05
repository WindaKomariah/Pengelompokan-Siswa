import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from kmodes.kprototypes import KPrototypes
from fpdf2 import FPDF  # <--- PERBAIKAN: Ganti 'fpdf' menjadi 'fpdf2'
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- KONSTANTA GLOBAL ---
# Warna (Pembaruan untuk palet yang lebih harmonis dan profesional)
PRIMARY_COLOR = "#2C2F7F"           # Biru Tua yang elegan
ACCENT_COLOR = "#7AA02F"            # Hijau Zaitun yang menenangkan (warna yang dipertahankan untuk judul)
BACKGROUND_COLOR = "#EAF0FA"        # Biru pucat yang nyaris putih
TEXT_COLOR = "#26272E"              # Abu-abu Tua gelap untuk teks, kontras tinggi
# HEADER_BACKGROUND_COLOR sekarang akan menggunakan ACCENT_COLOR untuk background hijau
HEADER_BACKGROUND_COLOR = ACCENT_COLOR # Menggunakan ACCENT_COLOR untuk latar belakang header
SIDEBAR_HIGHLIGHT_COLOR = "#4A5BAA" # Biru sedikit lebih terang untuk item aktif di sidebar
ACTIVE_BUTTON_BG_COLOR = "#3F51B5" # Biru Medium untuk latar belakang tombol aktif
ACTIVE_BUTTON_TEXT_COLOR = "#FFFFFF" # Teks putih
ACTIVE_BUTTON_BORDER_COLOR = "#FFD700" # Emas terang untuk border kiri (konsisten)

# Kolom Data yang Digunakan (Tidak berubah, ini sudah baik)
ID_COLS = ["No", "Nama", "JK", "Kelas"]
NUMERIC_COLS = ["Rata Rata Nilai Akademik", "Kehadiran"]
CATEGORICAL_COLS = ["Ekstrakurikuler Komputer", "Ekstrakurikuler Pertanian",
                    "Ekstrakurikuler Menjahit", "Ekstrakurikuler Pramuka"]
ALL_FEATURES_FOR_CLUSTERING = NUMERIC_COLS + CATEGORICAL_COLS

# --- CUSTOM CSS & HEADER ---
custom_css = f"""
<style>
    /* Global Reset and Spacing Adjustments */
    .stApp {{
        background-color: {BACKGROUND_COLOR};
        color: {TEXT_COLOR};
        font-family: 'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif;
    }}

    /* Reduce default margins and padding for the main content area */
    .main .block-container {{
        /* Sesuaikan padding-top untuk memberi ruang pada header sticky */
        padding-top: 7.5rem; /* Sesuaikan ini berdasarkan tinggi header Anda */
        padding-right: 4rem;
        padding-left: 4rem;
        padding-bottom: 3rem;
        max-width: 1200px;
        margin: auto;
    }}

    /* Target common wrapper divs that add vertical space more aggressively */
    [data-testid="stVerticalBlock"] > div:not(:last-child),
    [data-testid="stHorizontalBlock"] > div:not(:last-child) {{
        margin-bottom: 0.5rem !important;
        padding-bottom: 0px !important;
    }}
    .stVerticalBlock, .stHorizontalBlock {{
        gap: 1rem !important;
    }}

    /* Headings adjustments */
    h1, h2, h3, h4, h5, h6 {{
        margin-top: 1.5rem !important;
        margin-bottom: 0.8rem !important;
        padding-top: 0rem !important;
        padding-bottom: 0rem !important;
        color: {PRIMARY_COLOR};
        font-weight: 600;
    }}
    h1 {{ font-size: 2.5em; }}
    h2 {{ font-size: 2em; }}
    h3 {{ font-size: 1.5em; }}

    /* Specific for st.caption below the header */
    .stApp > div > div:first-child > div:nth-child(2) [data-testid="stText"] {{
        margin-top: 0.5rem !important;
        margin-bottom: 1rem !important;
        padding-top: 0 !important;
        padding-bottom: 0 !important;
        font-size: 0.95em;
        color: #666666;
    }}

    /* Target the first header/element in the main content area */
    .stApp > div > div:first-child > div:nth-child(3) h1:first-child,
    .stApp > div > div:first-child > div:nth-child(3) h2:first-child,
    .stApp > div > div:first-child > div:nth-child(3) h3:first-child
    {{
        margin-top: 1rem !important;
    }}
    .stApp > div > div:first-child > div:nth-child(3) [data-testid="stAlert"]:first-child {{
        margin-top: 1.2rem !important;
    }}

    /* Sidebar Styling */
    [data-testid="stSidebar"] {{
        background-color: {PRIMARY_COLOR};
        color: #ffffff;
        padding-top: 2.5rem;
    }}
    [data-testid="stSidebar"] * {{
        color: #ffffff;
    }}
    /* Style untuk tombol sidebar */
    [data-testid="stSidebar"] .stButton > button {{
        background-color: {PRIMARY_COLOR} !important;
        color: white !important;
        border: none !important;
        padding: 12px 25px !important;
        text-align: left !important;
        width: 100% !important;
        font-size: 17px !important;
        font-weight: 500 !important;
        margin: 0 !important;
        border-radius: 0 !important;
        transition: background-color 0.2s, color 0.2s, border-left 0.2s, box-shadow 0.2s;
        display: flex !important;
        justify-content: flex-start !important;
        align-items: center;
        gap: 10px;
    }}
    [data-testid="stSidebar"] .stButton > button:hover {{
        background-color: {SIDEBAR_HIGHLIGHT_COLOR} !important;
        color: #e0e0e0 !important;
    }}
    /* Mengurangi jarak vertikal antar tombol di sidebar */
    [data-testid="stSidebar"] [data-testid="stButton"] {{
        margin-bottom: 0px !important;
        padding: 0px !important;
    }}
    /* Pastikan tidak ada margin tambahan dari elemen parent */
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] > div {{
        margin-bottom: 0px !important;
    }}
    /* Active sidebar button styling */
    /* Target parent div dari button untuk class aktif yang persisten */
    [data-testid="stSidebar"] .st-sidebar-button-active {{
        background-color: {ACTIVE_BUTTON_BG_COLOR} !important; /* Warna latar belakang untuk item aktif */
        color: {ACTIVE_BUTTON_TEXT_COLOR} !important; /* Warna teks untuk item aktif */
        border-left: 6px solid {ACTIVE_BUTTON_BORDER_COLOR} !important; /* Border kiri yang menonjol */
        box-shadow: inset 4px 0 10px rgba(0,0,0,0.4) !important; /* Bayangan untuk kedalaman */
    }}
    /* Pastikan button di dalam active div juga mengikuti gaya aktif */
    [data-testid="stSidebar"] .st-sidebar-button-active > button {{
        background-color: {ACTIVE_BUTTON_BG_COLOR} !important;
        color: {ACTIVE_BUTTON_TEXT_COLOR} !important;
        font-weight: 700 !important; /* Teks lebih tebal untuk aktif */
    }}
    /* Non-active buttons maintain consistent border-left for visual alignment */
    [data-testid="stSidebar"] .stButton > button:not(.st-sidebar-button-active) {{
        border-left: 6px solid transparent !important;
        box-shadow: none !important;
    }}

    /* Custom Header - PERBAIKAN UTAMA DI SINI */
    .custom-header {{
        background-color: {HEADER_BACKGROUND_COLOR};
        padding: 25px 40px;
        color: white;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-radius: 0;
        box-shadow: 0 5px 15px rgba(0,0,0,0.25);
        
        /* Membuat header sticky */
        position: sticky;
        top: 0;
        left: 0;
        width: 100%;
        z-index: 1000; /* Pastikan di atas elemen lain saat scrolling */
        
        /* Hapus margin negatif yang menyebabkan masalah */
        margin: 0 !important;
    }}
    .custom-header h1 {{
        margin: 0 !important;
        font-size: 32px;
        font-weight: bold;
        color: white;
    }}
    .custom-header .kanan {{
        font-weight: 600;
        font-size: 19px;
        color: white;
        opacity: 0.9;
        text-align: right; /* Untuk memastikan teks tetap di kanan saat wrap */
    }}

    /* Media queries untuk responsivitas header di layar kecil (HP) */
    @media (max-width: 768px) {{
        .custom-header {{
            flex-direction: column; /* Tumpuk judul secara vertikal */
            align-items: flex-start; /* Sejajarkan ke kiri */
            padding: 15px 20px; /* Kurangi padding untuk layar kecil */
            text-align: left;
        }}
        .custom-header h1 {{
            font-size: 24px; /* Kecilkan ukuran font h1 */
            margin-bottom: 5px !important; /* Tambah sedikit margin di bawah h1 */
        }}
        .custom-header .kanan {{
            font-size: 14px; /* Kecilkan ukuran font teks kanan */
            text-align: left; /* Sesuaikan teks kanan ke kiri */
        }}
        .main .block-container {{
            padding-top: 10rem; /* Beri lebih banyak ruang di atas untuk header yang lebih tinggi */
            padding-right: 1rem;
            padding-left: 1rem;
        }}
    }}


    /* Alerts (Info, Success, Warning) */
    .stAlert {{
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 20px !important;
        margin-top: 20px !important;
        font-size: 0.95em;
        line-height: 1.5;
    }}
    .stAlert.info {{
        background-color: #e3f2fd;
        color: #1976D2;
        border-left: 6px solid #2196F3;
    }}
    .stAlert.success {{
        background-color: #e8f5e9;
        color: #388E3C;
        border-left: 6px solid #4CAF50;
    }}
    .stAlert.warning {{
        background-color: #fffde7;
        color: #FFA000;
        border-left: 6px solid #FFC107;
    }}
    .stAlert.error {{
        background-color: #ffebee;
        color: #D32F2F;
        border-left: 6px solid #F44336;
    }}

    /* Forms */
    .stForm {{
        background-color: white;
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin-top: 25px !important;
        margin-bottom: 25px !important;
        border: 1px solid #e0e0e0;
    }}

    /* Dataframe and Table styling - Increased margin for better separation */
    .stDataFrame, .stTable {{
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin-top: 30px !important;
        margin-bottom: 30px !important;
        border: 1px solid #e0e0e0;
    }}
    .stTable table th {{
        background-color: #f5f5f5 !important;
        color: {PRIMARY_COLOR} !important;
        font-weight: bold;
    }}
    .stTable table td {{
        padding: 8px 12px !important;
    }}

    /* Buttons (main content area) */
    .stButton > button {{
        background-color: {ACCENT_COLOR};
        color: white;
        padding: 10px 25px;
        border-radius: 8px;
        border: none;
        transition: background-color 0.2s ease-in-out, transform 0.1s ease-in-out;
        margin-top: 15px !important;
        margin-bottom: 8px !important;
        font-weight: 600;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }}
    .stButton > button:hover {{
        background-color: {PRIMARY_COLOR}; /* Ubah ke primary color saat hover */
        color: white;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.25);
    }}
    .stButton > button:active {{
        transform: translateY(0);
        box-shadow: 0 1px 3px rgba(0,0,0,0.2);
    }}

    /* Text Input & Number Input */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stDateInput > div > div > input,
    .stTimeInput > div > div > input {{
        border-radius: 8px;
        border: 1px solid #D1D1D1;
        padding: 10px 15px;
        margin-bottom: 8px !important;
        margin-top: 8px !important;
        background-color: white;
        box-shadow: inset 0 1px 3px rgba(0,0,0,0.05);
    }}
    .stTextInput label, .stNumberInput label, .stSelectbox label, .stCheckbox label, .stRadio label {{
        margin-bottom: 5px !important;
        padding-bottom: 0px !important;
        font-size: 0.98em;
        font-weight: 500;
        color: {TEXT_COLOR};
    }}

    /* Selectbox Styling (Input Box & Dropdown List) - PERBAIKAN LENGKAP */
    /* Target the main container of the selectbox input box */
    div[data-testid="stSelectbox"] > div:first-child {{
        width: 480px; /* Lebar yang lebih lebar untuk nama panjang */
        min-width: 300px; /* Pastikan tidak terlalu kecil */
    }}
    /* Target the actual visible button/input area of the selectbox input box */
    div[data-testid="stSelectbox"] > div > div > div > div[role="button"] {{
        width: 100% !important; /* Pastikan input mengisi lebar kontainer utama selectbox */
        white-space: normal; /* Mencegah teks terpotong ke baris baru */
        overflow: hidden; /* Sembunyikan jika teks terlalu panjang */
        text-overflow: ellipsis; /* Tambahkan elipsis jika disembunyikan */
        display: flex; /* Untuk memposisikan teks dengan baik */
        align-items: center; /* Pusatkan teks secara vertikal */
        height: auto; /* Izinkan tinggi menyesuaikan konten */
        box-sizing: border-box; /* Pastikan padding dan border termasuk dalam lebar/tinggi */
        padding-right: 35px; /* Tambahkan padding di kanan untuk memberi ruang pada panah dropdown */
    }}

    /* Target the dropdown caret (panah) itself to ensure it's positioned correctly */
    /* Streamlit sering mengubah nama kelas internalnya.
        Beberapa selector umum disertakan. Anda mungkin perlu memeriksa elemen di browser Anda
        (klik kanan > inspect) untuk class/data-testid yang tepat untuk ikon panah pada versi Streamlit Anda. */
    div[data-testid="stSelectbox"] .st-bh .st-cj, /* Selector umum yang mungkin bekerja */
    div[data-testid="stSelectbox"] .st-ck .st-ci, /* Selector lain yang mungkin bekerja */
    div[data-testid="stSelectbox"] [data-testid="stFormSubmitButton"] + div > div > div > button > svg /* Selector lebih spesifik jika panah adalah SVG di tombol */
    {{
        position: absolute; /* Posisikan secara absolut */
        right: 10px; /* Sesuaikan jarak dari kanan */
        top: 50%; /* Pusatkan secara vertikal */
        transform: translateY(-50%); /* Penyesuaian vertikal */
        pointer-events: none; /* Pastikan tidak mengganggu klik pada selectbox */
        z-index: 1; /* Pastikan di atas elemen lain jika ada tumpang tindih */
    }}

    /* NEW: Styling for the dropdown list (pop-up) itself that appears below the selectbox */
    /* This targets the container that holds the dropdown options. Using role="listbox" is robust. */
    div[role="listbox"][aria-orientation="vertical"] {{
        width: 500px !important; /* Set a fixed width for the dropdown list, slightly more than input */
        max-width: 600px !important; /* Batasi lebar maksimum agar tidak terlalu besar */
        min-width: 400px !important; /* Pastikan tidak terlalu sempit */
        overflow-x: hidden !important; /* Sembunyikan scroll horizontal */
        overflow-y: auto !important; /* Aktifkan scroll vertikal untuk daftar panjang */
        box-sizing: border-box; /* Pastikan padding/border dihitung dalam lebar total */
        border-radius: 8px; /* Konsistenkan border radius */
        border: 1px solid #D1D1D1; /* Tambahkan border */
        box-shadow: 0 4px 12px rgba(0,0,0,0.1); /* Tambahkan shadow */
        background-color: white; /* Pastikan background putih */
    }}

    /* NEW: Ensure individual options within the dropdown list stretch and wrap correctly */
    div[role="option"] {{ /* Individual options within the listbox */
        white-space: normal !important; /* Izinkan teks membungkus */
        word-wrap: break-word !important; /* Memecah kata yang panjang */
        padding-right: 15px !important; /* Tambahkan padding di kanan untuk teks */
        padding-left: 15px !important; /* Tambahkan padding di kiri untuk teks */
        line-height: 1.4; /* Meningkatkan jarak antar baris teks jika membungkus */
        min-height: 38px; /* Memberi tinggi minimum untuk setiap opsi */
        display: flex; /* Untuk aligment vertikal */
        align-items: center; /* Pusatkan teks secara vertikal di dalam opsi */
    }}
    /* Hover state for dropdown options */
    div[role="option"]:hover {{
        background-color: #e0e0e0; /* Warna latar belakang saat di-hover */
        color: {PRIMARY_COLOR};
    }}


    /* Scrollbar Styling - Diperbarui agar lebih jelas dan tidak terpotong */
    ::-webkit-scrollbar {{
        width: 10px; /* Lebar scrollbar yang sedikit lebih besar */
    }}
    ::-webkit-scrollbar-thumb {{
        background: {ACCENT_COLOR};
        border-radius: 5px; /* Radius yang sedikit lebih besar */
    }}
    ::-webkit-scrollbar-track {{
        background: #e9e9e9;
    }}


    /* Checkbox & Radio */
    .stCheckbox label, .stRadio label {{
        display: flex;
        align-items: center;
        cursor: pointer;
        user-select: none;
    }}
    .stCheckbox {{
        margin-bottom: 10px !important;
        margin-top: 10px !important;
    }}

    /* Expander styling */
    .stExpander {{
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }}
    .stExpander > div > div > p {{
        font-weight: 600;
        color: {PRIMARY_COLOR};
    }}

    /* Columns spacing */
    div[data-testid="column"] {{
        gap: 2rem;
    }}
    
    /* Overall top padding adjustment for main block (Streamlit's main content wrapper) */
    /* Ini seharusnya tidak perlu lagi karena padding-top sudah di .main .block-container */
    /* .css-1d3fclg.eggyngi2 {{
        padding-top: 1rem !important;
    }} */

    /* Ensure specific elements have appropriate top margins after the main header */
    .stApp > div > div:first-child > div:nth-child(3) > div:first-child {{
        /* Mengurangi margin top karena padding sudah diatur pada block-container */
        margin-top: 0rem !important;
    }}
</style>
"""

# Header HTML yang lebih menarik
# Tidak ada perubahan pada header_html karena kita akan mengontrol tata letak via CSS
header_html = f"""
<div class="custom-header">
    <div><h1>PENGELOMPOKAN SISWA</h1></div>
    <div class="kanan">MADRASAH ALIYAH AL-HIKMAH</div>
</div>
"""

# Konfigurasi halaman Streamlit
st.set_page_config(page_title="Klasterisasi K-Prototype Siswa", layout="wide", initial_sidebar_state="expanded")
st.markdown(custom_css, unsafe_allow_html=True)
st.markdown(header_html, unsafe_allow_html=True)

# Hapus spasi vertikal tambahan ini karena padding-top di .main .block-container sudah menangani
# st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)

# --- INISIALISASI SESSION STATE ---
if 'df_original' not in st.session_state:
    st.session_state.df_original = None
if 'df_preprocessed_for_clustering' not in st.session_state:
    st.session_state.df_preprocessed_for_clustering = None
if 'df_clustered' not in st.session_state:
    st.session_state.df_clustered = None

if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'kproto_model' not in st.session_state:
    st.session_state.kproto_model = None
if 'categorical_features_indices' not in st.session_state:
    st.session_state.categorical_features_indices = None
if 'n_clusters' not in st.session_state:
    st.session_state.n_clusters = 3 # Default jumlah klaster
if 'cluster_characteristics_map' not in st.session_state:
    st.session_state.cluster_characteristics_map = {}

# --- FUNGSI PEMBANTU ---

def generate_pdf_profil_siswa(nama, data_siswa_dict, klaster, cluster_desc_map):
    """
    Menghasilkan laporan PDF profil siswa.
    """
    # Menggunakan fpdf2 dan menambahkan font yang mendukung Unicode
    # PERBAIKAN: Ganti FPDF dengan FPDF2
    pdf = FPDF(orientation='P', unit='mm', format='A4')
    pdf.add_page()
    
    # PERBAIKAN: Gunakan font yang mendukung UTF-8
    # Agar aman, kita akan menambahkan font yang umum dan mendukung Unicode.
    # Namun, agar kode ini bisa berjalan tanpa file font terpisah,
    # kita akan menggunakan font bawaan yang mendukung beberapa karakter.
    # Jika masih ada error, pengguna perlu menyediakan file font seperti 'DejaVuSans.ttf'
    # dan menambahkannya dengan pdf.add_font(). Untuk saat ini, kita coba dengan Arial.

    # Menyiapkan font untuk judul
    pdf.set_font("Arial", "B", 16)
    pdf.set_text_color(44, 47, 127) # Warna biru tua

    # Menggunakan multi_cell untuk judul agar lebih fleksibel
    pdf.multi_cell(0, 10, "PROFIL SISWA - HASIL KLASTERISASI", align='C')
    pdf.ln(10)

    # Menyiapkan font untuk konten
    pdf.set_font("Arial", "", 10)
    pdf.set_text_color(0, 0, 0)
    keterangan_umum = (
        "Laporan ini menyajikan profil detail siswa berdasarkan hasil pengelompokan "
        "menggunakan Algoritma K-Prototype. Klasterisasi dilakukan berdasarkan "
        "nilai akademik, kehadiran, dan partisipasi ekstrakurikuler siswa. "
        "Informasi klaster ini dapat digunakan untuk memahami kebutuhan siswa dan "
        "merancang strategi pembinaan yang sesuai."
    )
    pdf.multi_cell(0, 5, keterangan_umum, align='J')
    pdf.ln(5)

    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 8, f"Nama Siswa: {nama}", ln=True)
    pdf.cell(0, 8, f"Klaster Hasil: {klaster}", ln=True)
    pdf.ln(3)

    klaster_desc = cluster_desc_map.get(klaster, "Deskripsi klaster tidak tersedia.")
    # PERBAIKAN: Pastikan deskripsi klaster juga ditangani dengan multi_cell
    pdf.set_font("Arial", "I", 10)
    pdf.set_text_color(80, 80, 80)
    pdf.multi_cell(0, 5, f"Karakteristik Klaster {klaster}: {klaster_desc}", align='J')
    pdf.ln(5)

    pdf.set_font("Arial", "", 10)
    pdf.set_text_color(0, 0, 0)

    ekskul_diikuti = []
    ekskul_cols_full_names = ["Ekstrakurikuler Komputer", "Ekstrakurikuler Pertanian", "Ekstrakurikuler Menjahit", "Ekstrakurikuler Pramuka"]
    for col in ekskul_cols_full_names:
        # Periksa apakah key ada dan nilainya 1 (sesuai data biner 0/1)
        if data_siswa_dict.get(col) == 1:
            ekskul_diikuti.append(col.replace("Ekstrakurikuler ", ""))

    display_data = {
        "Nomor Induk": data_siswa_dict.get("No", "-"),
        "Jenis Kelamin": data_siswa_dict.get("JK", "-"),
        "Kelas": data_siswa_dict.get("Kelas", "-"),
        "Rata-rata Nilai Akademik": f"{data_siswa_dict.get('Rata Rata Nilai Akademik', '-'):.2f}",
        "Persentase Kehadiran": f"{data_siswa_dict.get('Kehadiran', '-'):.2%}",
        "Ekstrakurikuler yang Diikuti": ", ".join(ekskul_diikuti) if ekskul_diikuti else "Tidak mengikuti ekstrakurikuler",
    }

    for key, val in display_data.items():
        pdf.cell(0, 7, f"{key}: {val}", ln=True)

    # BARIS INI YANG DIUBAH: MENAMBAHKAN .encode('utf-8')
    # PERBAIKAN: Dengan fpdf2, .encode('utf-8') tidak lagi diperlukan pada output.
    # Output stream sudah berupa bytes.
    return pdf.output(dest='S')

def preprocess_data(df):
    """
    Melakukan praproses data: membersihkan kolom, mengubah tipe data kategorikal,
    dan melakukan normalisasi Z-score pada kolom numerik.
    Mengembalikan dataframe yang sudah diproses dan scaler yang digunakan.
    """
    df_processed = df.copy()

    # Periksa dan bersihkan nama kolom dari spasi tambahan atau karakter non-ASCII
    df_processed.columns = [col.strip() for col in df_processed.columns]

    # Pastikan kolom yang dibutuhkan ada
    missing_cols = [col for col in NUMERIC_COLS + CATEGORICAL_COLS if col not in df_processed.columns]
    if missing_cols:
        st.error(f"Kolom-kolom berikut tidak ditemukan dalam data Anda: {', '.join(missing_cols)}. Harap periksa file Excel Anda dan pastikan nama kolom sudah benar.")
        return None, None

    # Hapus kolom identitas untuk klasterisasi
    df_clean_for_clustering = df_processed.drop(columns=ID_COLS, errors="ignore")

    # Konversi kolom kategorikal ke tipe string untuk KPrototypes
    for col in CATEGORICAL_COLS:
        # Tangani nilai NaN pada kolom kategorikal, isi dengan string '0' (atau mode)
        df_clean_for_clustering[col] = df_clean_for_clustering[col].fillna(0).astype(str)

    # Tangani nilai NaN pada kolom numerik, isi dengan rata-rata kolom
    for col in NUMERIC_COLS:
        if df_clean_for_clustering[col].isnull().any():
            mean_val = df_clean_for_clustering[col].mean()
            df_clean_for_clustering[col] = df_clean_for_clustering[col].fillna(mean_val)
            st.warning(f"Nilai kosong pada kolom '{col}' diisi dengan rata-rata: {mean_val:.2f}.")

    # Normalisasi Z-score untuk kolom numerik
    scaler = StandardScaler()
    df_clean_for_clustering[NUMERIC_COLS] = scaler.fit_transform(df_clean_for_clustering[NUMERIC_COLS])

    return df_clean_for_clustering, scaler

def run_kprototypes_clustering(df_preprocessed, n_clusters):
    """
    Menjalankan algoritma K-Prototypes pada data yang telah diproses.
    Mengembalikan dataframe dengan kolom klaster, model kproto, dan indeks kolom kategorikal.
    """
    df_for_clustering = df_preprocessed.copy()

    # Siapkan data untuk KPrototypes
    X_data = df_for_clustering[ALL_FEATURES_FOR_CLUSTERING]
    X = X_data.to_numpy()

    # Dapatkan indeks kolom kategorikal
    categorical_feature_indices = [X_data.columns.get_loc(c) for c in CATEGORICAL_COLS]

    # Inisialisasi dan latih model KPrototypes
    try:
        kproto = KPrototypes(n_clusters=n_clusters, init='Huang', n_init=10, verbose=0, random_state=42, n_jobs=-1) # Gunakan semua CPU
        clusters = kproto.fit_predict(X, categorical=categorical_feature_indices)
    except Exception as e:
        st.error(f"Terjadi kesalahan saat menjalankan K-Prototypes: {e}. Pastikan data Anda cukup bervariasi untuk jumlah klaster yang dipilih.")
        return None, None, None

    df_for_clustering["Klaster"] = clusters
    return df_for_clustering, kproto, categorical_feature_indices

def generate_cluster_descriptions(df_clustered, n_clusters, numeric_cols, categorical_cols):
    """
    Menghasilkan deskripsi karakteristik untuk setiap klaster.
    """
    cluster_characteristics_map = {}
    
    # Ambil nilai min/max asli untuk normalisasi balik deskripsi
    df_original_numeric = st.session_state.df_original[NUMERIC_COLS]
    original_min_vals = df_original_numeric.min()
    original_max_vals = df_original_numeric.max()
    original_mean_vals = df_original_numeric.mean()
    original_std_vals = df_original_numeric.std()

    for i in range(n_clusters):
        cluster_data = df_clustered[df_clustered["Klaster"] == i]

        # Rata-rata untuk fitur numerik (setelah normalisasi)
        avg_scaled_values = cluster_data[numeric_cols].mean()
        # Modus untuk fitur kategorikal
        mode_values = cluster_data[categorical_cols].mode().iloc[0]

        desc = ""
        
        # Lebih sederhana, bandingkan dengan 0 (rata-rata setelah Z-score)
        # Deskripsi nilai akademik
        if avg_scaled_values["Rata Rata Nilai Akademik"] > 0.75:
            desc += "Siswa di klaster ini memiliki nilai akademik cenderung sangat tinggi. "
        elif avg_scaled_values["Rata Rata Nilai Akademik"] > 0.25:
            desc += "Siswa di klaster ini memiliki nilai akademik cenderung di atas rata-rata. "
        elif avg_scaled_values["Rata Rata Nilai Akademik"] < -0.75:
            desc += "Siswa di klaster ini memiliki nilai akademik cenderung sangat rendah. "
        elif avg_scaled_values["Rata Rata Nilai Akademik"] < -0.25:
            desc += "Siswa di klaster ini memiliki nilai akademik cenderung di bawah rata-rata. "
        else:
            desc += "Siswa di klaster ini memiliki nilai akademik cenderung rata-rata. "

        # Deskripsi kehadiran
        if avg_scaled_values["Kehadiran"] > 0.75:
            desc += "Tingkat kehadiran cenderung sangat tinggi. "
        elif avg_scaled_values["Kehadiran"] > 0.25:
            desc += "Tingkat kehadiran cenderung di atas rata-rata. "
        elif avg_scaled_values["Kehadiran"] < -0.75:
            desc += "Tingkat kehadiran cenderung sangat rendah. "
        elif avg_scaled_values["Kehadiran"] < -0.25:
            desc += "Tingkat kehadiran cenderung di bawah rata-rata. "
        else:
            desc += "Tingkat kehadiran cenderung rata-rata. "

        # Deskripsi ekstrakurikuler
        ekskul_aktif_modes = [col_name for col_name in categorical_cols if mode_values[col_name] == '1']
        if ekskul_aktif_modes:
            desc += f"Siswa di klaster ini aktif dalam ekstrakurikuler: {', '.join([c.replace('Ekstrakurikuler ', '') for c in ekskul_aktif_modes])}."
        else:
            desc += "Siswa di klaster ini kurang aktif dalam kegiatan ekstrakurikuler."

        cluster_characteristics_map[i] = desc
    return cluster_characteristics_map

# --- NAVIGASI SIDEBAR ---
st.sidebar.title("MENU NAVIGASI")
st.sidebar.markdown("---")

menu_options = [
    "Unggah Data",
    "Praproses & Normalisasi Data",
    "Klasterisasi Data K-Prototypes",
    "Prediksi Klaster Siswa Baru",
    "Visualisasi & Profil Klaster",
    "Lihat Profil Siswa Individual"
]

# Inisialisasi 'current_menu' jika belum ada
if 'current_menu' not in st.session_state:
    st.session_state.current_menu = menu_options[0]

# Tampilan tombol sidebar dengan penanda aktif yang lebih baik
for option in menu_options:
    icon_map = {
        "Unggah Data": "⬆",
        "Praproses & Normalisasi Data": "⚙",
        "Klasterisasi Data K-Prototypes": "📊",
        "Prediksi Klaster Siswa Baru": "🔮",
        "Visualisasi & Profil Klaster": "📈",
        "Lihat Profil Siswa Individual": "👤"
    }
    display_name = f"{icon_map.get(option, '')} {option}"
    
    # Gunakan kunci unik untuk setiap tombol
    button_key = f"nav_button_{option.replace(' ', '_').replace('&', 'and')}"

    if st.sidebar.button(display_name, key=button_key):
        st.session_state.current_menu = option
        st.rerun()

# --- JavaScript untuk Menandai Halaman Aktif di Sidebar (Inject sekali, setelah semua tombol dirender) ---
js_highlight_active_button = f"""
<script>
    // Fungsi untuk membersihkan teks tombol dari emoji dan spasi ekstra
    function cleanButtonText(text) {{
        return (text || '').replace(/\\p{{Emoji}}/gu, '').trim();
    }}

    // Fungsi untuk menandai tombol sidebar aktif
    function highlightActiveSidebarButton() {{
        var currentMenu = '{st.session_state.current_menu}'; // Ambil menu aktif dari Python
        var cleanCurrentMenuName = cleanButtonText(currentMenu);

        var sidebarButtonContainers = window.parent.document.querySelectorAll('[data-testid="stSidebar"] [data-testid="stButton"]');
        
        sidebarButtonContainers.forEach(function(container) {{
            var button = container.querySelector('button');
            if (button) {{
                var buttonText = cleanButtonText(button.innerText || button.textContent);
                
                container.classList.remove('st-sidebar-button-active');

                if (buttonText === cleanCurrentMenuName) {{
                    container.classList.add('st-sidebar-button-active');
                }}
            }}
        }});
    }}

    const observer = new MutationObserver((mutationsList, observer) => {{
        const sidebarChanged = mutationsList.some(mutation =>
            mutation.target.closest('[data-testid="stSidebar"]')
        );
        if (sidebarChanged) {{
            highlightActiveSidebarButton();
        }}
    }});

    observer.observe(window.parent.document.body, {{ childList: true, subtree: true }});

    highlightActiveSidebarButton();
</script>
"""
if hasattr(st, 'html'):
    st.html(js_highlight_active_button)
else:
    st.markdown(js_highlight_active_button, unsafe_allow_html=True)


# --- KONTEN HALAMAN UTAMA BERDASARKAN MENU TERPILIH ---

# Gunakan sebuah div untuk mengatur jarak antara header global dan konten setiap halaman
# st.markdown("<div id='page-top-spacer' style='margin-top: 30px;'></div>", unsafe_allow_html=True)
# Spacer ini tidak lagi diperlukan karena padding-top pada .main .block-container sudah menangani

if st.session_state.current_menu == "Unggah Data":
    st.header("Unggah Data Siswa")
    st.markdown("""
    <div style='background-color:#e3f2fd; padding:15px; border-radius:10px; border-left: 5px solid #2196F3;'>
    Silakan unggah file Excel (.xlsx) yang berisi dataset siswa. Pastikan file Anda memiliki
    kolom-kolom berikut agar sistem dapat bekerja dengan baik:<br><br>
    <ul>
        <li><b>Kolom Identitas:</b> "No", "Nama", "JK", "Kelas"</li>
        <li><b>Kolom Numerik (untuk analisis):</b> "Rata Rata Nilai Akademik", "Kehadiran"</li>
        <li><b>Kolom Kategorikal (untuk analisis, nilai 0 atau 1):</b> "Ekstrakurikuler Komputer", "Ekstrakurikuler Pertanian", "Ekstrakurikuler Menjahit", "Ekstrakurikuler Pramuka"</li>
    </ul>
    Pastikan nama kolom sudah persis sama dan tidak ada kesalahan penulisan.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---") # Visual separator
    
    uploaded_file = st.file_uploader("Pilih File Excel Dataset", type=["xlsx"], help="Unggah file Excel Anda di sini. Hanya format .xlsx yang didukung.")
    if uploaded_file:
        try:
            df = pd.read_excel(uploaded_file,
                                engine='openpyxl')
            st.session_state.df_original = df
            st.success("Data berhasil diunggah! Anda dapat melanjutkan ke langkah praproses.")
            st.subheader("Preview Data yang Diunggah:")
            st.dataframe(df, use_container_width=True, height=300)

        except Exception as e:
            st.error(f"Terjadi kesalahan saat membaca file: {e}")
            st.session_state.df_original = None

elif st.session_state.current_menu == "Praproses & Normalisasi Data":
    st.header("Praproses dan Normalisasi Data")
    st.markdown("Langkah ini akan membersihkan data, mengisi nilai yang hilang, dan menormalisasi fitur numerik agar siap untuk klasterisasi.")
    
    if st.session_state.df_original is None:
        st.warning("Silakan unggah data terlebih dahulu pada menu 'Unggah Data'.")
    else:
        if st.button("Lakukan Praproses Data"):
            with st.spinner("Memproses data..."):
                df_preprocessed, scaler = preprocess_data(st.session_state.df_original)
                if df_preprocessed is not None:
                    st.session_state.df_preprocessed_for_clustering = df_preprocessed
                    st.session_state.scaler = scaler
                    st.success("Praproses data berhasil! Data siap untuk diklasterisasi.")
                    st.subheader("Preview Data yang Telah Diproses:")
                    st.dataframe(df_preprocessed, use_container_width=True, height=300)

elif st.session_state.current_menu == "Klasterisasi Data K-Prototypes":
    st.header("Klasterisasi Data K-Prototypes")
    st.markdown("Pilih jumlah klaster yang Anda inginkan, lalu jalankan algoritma K-Prototypes.")

    if st.session_state.df_preprocessed_for_clustering is None:
        st.warning("Silakan selesaikan langkah 'Unggah Data' dan 'Praproses Data' terlebih dahulu.")
    else:
        # Pilihan jumlah klaster
        n_clusters_input = st.slider(
            "Pilih Jumlah Klaster (k)",
            min_value=2,
            max_value=10,
            value=st.session_state.n_clusters,
            step=1,
            help="Jumlah klaster optimal dapat bervariasi. Coba beberapa nilai berbeda untuk menemukan hasil terbaik."
        )
        st.session_state.n_clusters = n_clusters_input

        if st.button("Jalankan Klasterisasi"):
            with st.spinner("Menjalankan K-Prototypes..."):
                df_result, kproto_model, cat_indices = run_kprototypes_clustering(
                    st.session_state.df_preprocessed_for_clustering,
                    st.session_state.n_clusters
                )
                if df_result is not None:
                    # Gabungkan klaster hasil dengan data original
                    df_final = st.session_state.df_original.copy()
                    df_final["Klaster"] = df_result["Klaster"]
                    st.session_state.df_clustered = df_final
                    st.session_state.kproto_model = kproto_model
                    st.session_state.categorical_features_indices = cat_indices

                    # Hasilkan deskripsi klaster
                    st.session_state.cluster_characteristics_map = generate_cluster_descriptions(
                        df_final, st.session_state.n_clusters, NUMERIC_COLS, CATEGORICAL_COLS
                    )

                    st.success("Klasterisasi berhasil!")
                    st.subheader("Hasil Klasterisasi (Preview):")
                    st.dataframe(df_final.head(), use_container_width=True, height=200)

elif st.session_state.current_menu == "Prediksi Klaster Siswa Baru":
    st.header("Prediksi Klaster Siswa Baru")
    st.markdown("Gunakan model yang sudah dilatih untuk memprediksi klaster siswa baru.")

    if st.session_state.kproto_model is None:
        st.warning("Harap selesaikan langkah 'Klasterisasi Data' terlebih dahulu untuk melatih model.")
    else:
        st.info("Masukkan data siswa baru di bawah ini:")
        
        with st.form("input_siswa_baru"):
            st.subheader("Data Akademik dan Kehadiran")
            new_akademik = st.number_input(
                "Rata-rata Nilai Akademik", min_value=0.0, max_value=100.0, value=75.0, step=0.1
            )
            new_kehadiran = st.number_input(
                "Kehadiran (%)", min_value=0.0, max_value=100.0, value=90.0, step=0.1
            )

            st.subheader("Keikutsertaan Ekstrakurikuler")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                new_ekskul_komputer = st.checkbox("Komputer")
            with col2:
                new_ekskul_pertanian = st.checkbox("Pertanian")
            with col3:
                new_ekskul_menjahit = st.checkbox("Menjahit")
            with col4:
                new_ekskul_pramuka = st.checkbox("Pramuka")

            submitted = st.form_submit_button("Prediksi Klaster")

        if submitted:
            # Siapkan data siswa baru
            new_data = pd.DataFrame({
                "Rata Rata Nilai Akademik": [new_akademik],
                "Kehadiran": [new_kehadiran],
                "Ekstrakurikuler Komputer": [1 if new_ekskul_komputer else 0],
                "Ekstrakurikuler Pertanian": [1 if new_ekskul_pertanian else 0],
                "Ekstrakurikuler Menjahit": [1 if new_ekskul_menjahit else 0],
                "Ekstrakurikuler Pramuka": [1 if new_ekskul_pramuka else 0],
            })
            
            # Praproses data baru
            new_data_preprocessed = new_data.copy()
            new_data_preprocessed[NUMERIC_COLS] = st.session_state.scaler.transform(new_data_preprocessed[NUMERIC_COLS])
            new_data_preprocessed[CATEGORICAL_COLS] = new_data_preprocessed[CATEGORICAL_COLS].astype(str)

            # Lakukan prediksi
            new_data_array = new_data_preprocessed.to_numpy()
            predicted_cluster = st.session_state.kproto_model.predict(new_data_array, categorical=st.session_state.categorical_features_indices)

            st.subheader("Hasil Prediksi:")
            st.success(f"Siswa ini kemungkinan besar termasuk dalam **Klaster {predicted_cluster[0]}**.")
            
            cluster_desc = st.session_state.cluster_characteristics_map.get(
                predicted_cluster[0], "Deskripsi klaster tidak tersedia."
            )
            st.markdown(f"**Karakteristik Klaster:** {cluster_desc}")


elif st.session_state.current_menu == "Visualisasi & Profil Klaster":
    st.header("Visualisasi dan Profil Klaster")
    st.markdown("Lihat karakteristik setiap klaster untuk mendapatkan wawasan lebih dalam.")

    if st.session_state.df_clustered is None:
        st.warning("Silakan selesaikan langkah 'Klasterisasi Data' terlebih dahulu.")
    else:
        st.subheader("Karakteristik Klaster Berdasarkan Fitur")
        
        cluster_summary = st.session_state.df_clustered.groupby("Klaster").agg(
            {
                "Rata Rata Nilai Akademik": ["mean"],
                "Kehadiran": ["mean"],
                "Ekstrakurikuler Komputer": ["sum"],
                "Ekstrakurikuler Pertanian": ["sum"],
                "Ekstrakurikuler Menjahit": ["sum"],
                "Ekstrakurikuler Pramuka": ["sum"],
            }
        ).reset_index()
        st.dataframe(cluster_summary, use_container_width=True)

        st.markdown("---")
        st.subheader("Visualisasi Klaster")
        
        # Plot Rata-rata Nilai Akademik
        fig_nilai = plt.figure(figsize=(8, 5))
        sns.barplot(x="Klaster", y=("Rata Rata Nilai Akademik", "mean"), data=cluster_summary, palette="viridis")
        plt.title("Rata-rata Nilai Akademik per Klaster")
        plt.xlabel("Klaster")
        plt.ylabel("Rata-rata Nilai")
        st.pyplot(fig_nilai)

        # Plot Rata-rata Kehadiran
        fig_kehadiran = plt.figure(figsize=(8, 5))
        sns.barplot(x="Klaster", y=("Kehadiran", "mean"), data=cluster_summary, palette="magma")
        plt.title("Rata-rata Kehadiran per Klaster")
        plt.xlabel("Klaster")
        plt.ylabel("Rata-rata Kehadiran (%)")
        st.pyplot(fig_kehadiran)

        # Plot Jumlah Partisipasi Ekstrakurikuler
        st.subheader("Partisipasi Ekstrakurikuler per Klaster")
        ekskul_summary = cluster_summary.melt(id_vars="Klaster", value_vars=[
            ("Ekstrakurikuler Komputer", "sum"),
            ("Ekstrakurikuler Pertanian", "sum"),
            ("Ekstrakurikuler Menjahit", "sum"),
            ("Ekstrakurikuler Pramuka", "sum")
        ], var_name="Ekstrakurikuler", value_name="Jumlah Siswa")
        ekskul_summary["Ekstrakurikuler"] = ekskul_summary["Ekstrakurikuler"].apply(lambda x: x[0].replace("Ekstrakurikuler ", ""))

        fig_ekskul = plt.figure(figsize=(10, 6))
        sns.barplot(
            data=ekskul_summary,
            x="Ekstrakurikuler",
            y="Jumlah Siswa",
            hue="Klaster",
            palette="plasma",
            order=ekskul_summary.groupby("Ekstrakurikuler")["Jumlah Siswa"].sum().sort_values(ascending=False).index
        )
        plt.title("Jumlah Siswa yang Mengikuti Ekstrakurikuler per Klaster")
        plt.xlabel("Jenis Ekstrakurikuler")
        plt.ylabel("Jumlah Siswa")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig_ekskul)

        # Deskripsi tekstual klaster
        st.subheader("Deskripsi Karakteristik Klaster")
        for klaster_id, desc in st.session_state.cluster_characteristics_map.items():
            st.info(f"**Klaster {klaster_id}:** {desc}")


elif st.session_state.current_menu == "Lihat Profil Siswa Individual":
    st.header("Lihat Profil Siswa Individual")
    st.markdown("Pilih nama siswa untuk melihat detail profil dan laporan klasterisasinya.")

    if st.session_state.df_clustered is None:
        st.warning("Silakan selesaikan langkah 'Klasterisasi Data' terlebih dahulu.")
    else:
        nama_siswa_list = st.session_state.df_clustered["Nama"].unique()
        selected_nama = st.selectbox(
            "Pilih Nama Siswa",
            options=["-- Pilih Siswa --"] + sorted(nama_siswa_list),
            help="Cari dan pilih nama siswa untuk melihat profilnya."
        )

        if selected_nama != "-- Pilih Siswa --":
            data_siswa = st.session_state.df_clustered[st.session_state.df_clustered["Nama"] == selected_nama].iloc[0]
            data_siswa_dict = data_siswa.to_dict()

            st.subheader(f"Profil Siswa: {data_siswa_dict['Nama']}")
            klaster_siswa = data_siswa_dict["Klaster"]
            cluster_desc = st.session_state.cluster_characteristics_map.get(klaster_siswa, "Deskripsi klaster tidak tersedia.")

            st.write(f"**Klaster Hasil:** **`Klaster {klaster_siswa}`**")
            st.markdown(f"**Karakteristik Klaster:** {cluster_desc}")
            
            st.markdown("---")
            st.write("### Data Lengkap Siswa")
            
            col_id, col_akademik = st.columns(2)
            with col_id:
                st.write(f"**Nomor:** {data_siswa_dict.get('No', '-')}")
                st.write(f"**Jenis Kelamin:** {data_siswa_dict.get('JK', '-')}")
                st.write(f"**Kelas:** {data_siswa_dict.get('Kelas', '-')}")
            with col_akademik:
                st.write(f"**Rata-rata Nilai Akademik:** {data_siswa_dict.get('Rata Rata Nilai Akademik', '-'):.2f}")
                st.write(f"**Persentase Kehadiran:** {data_siswa_dict.get('Kehadiran', '-'):.2f}%")

            ekskul_diikuti = [
                col.replace("Ekstrakurikuler ", "") for col in CATEGORICAL_COLS
                if data_siswa_dict.get(col) == 1
            ]
            st.write(f"**Ekstrakurikuler yang Diikuti:** {', '.join(ekskul_diikuti) if ekskul_diikuti else 'Tidak mengikuti ekstrakurikuler'}")
            
            # Tombol untuk generate PDF
            pdf_data_bytes = generate_pdf_profil_siswa(
                selected_nama,
                data_siswa_dict,
                klaster_siswa,
                st.session_state.cluster_characteristics_map
            )
            
            st.download_button(
                label="Unduh Laporan PDF",
                data=pdf_data_bytes,
                file_name=f"Profil_Siswa_{selected_nama.replace(' ', '_')}.pdf",
                mime="application/pdf",
                help="Klik untuk mengunduh laporan PDF profil siswa ini."
            )
