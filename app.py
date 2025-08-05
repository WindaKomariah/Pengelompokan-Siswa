import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from kmodes.kprototypes import KPrototypes
from fpdf import FPDF
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- KONSTANTA GLOBAL ---
# Warna (Pembaruan untuk palet yang lebih harmonis dan profesional)
PRIMARY_COLOR = "#2C2F7F"           # Biru Tua yang elegan
ACCENT_COLOR = "#7AA02F"            # Hijau Zaitun yang menenangkan (warna yang dipertahankan untuk judul)
BACKGROUND_COLOR = "#EAF0FA"        # Biru pucat yang nyaris putih
TEXT_COLOR = "#26272E"              # Abu-abu Tua gelap untuk teks, kontras tinggi
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
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.set_text_color(44, 47, 127) # Warna biru tua

    pdf.cell(0, 10, "PROFIL SISWA - HASIL KLASTERISASI", ln=True, align='C')
    pdf.ln(10)

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

    # --- PERBAIKAN PENTING DI SINI ---
    output_pdf = pdf.output(dest='S')
    if isinstance(output_pdf, str):
        return output_pdf.encode('utf-8')
    else:
        return output_pdf

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
            st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True) # Spasi setelah dataframe
        except Exception as e:
            st.error(f"Terjadi kesalahan saat membaca file: {e}. Pastikan format file Excel benar dan tidak rusak.")

elif st.session_state.current_menu == "Praproses & Normalisasi Data":
    st.header("Praproses Data & Normalisasi Z-score")
    if st.session_state.df_original is None or st.session_state.df_original.empty:
        st.warning("Silakan unggah data terlebih dahulu di menu 'Unggah Data'.")
    else:
        st.markdown("""
        <div style='background-color:#e3f2fd; padding:15px; border-radius:10px; border-left: 5px solid #2196F3;'>
        Pada tahap ini, data akan disiapkan untuk analisis klasterisasi. Proses yang dilakukan meliputi:
        <ul>
            <li><b>Pembersihan Data:</b> Menangani nilai-nilai yang hilang (missing values) pada kolom numerik (diisi dengan rata-rata).</li>
            <li><b>Konversi Tipe Data:</b> Memastikan kolom kategorikal memiliki tipe data yang sesuai untuk algoritma.</li>
            <li><b>Normalisasi Z-score:</b> Mengubah skala fitur numerik (nilai akademik & kehadiran) agar memiliki rata-rata nol dan deviasi standar satu, sehingga semua fitur memiliki bobot yang setara dalam perhitungan klasterisasi.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        if st.button("Jalankan Praproses & Normalisasi"):
            with st.spinner("Sedang memproses dan menormalisasi data..."):
                df_preprocessed, scaler = preprocess_data(st.session_state.df_original)

            if df_preprocessed is not None and scaler is not None:
                st.session_state.df_preprocessed_for_clustering = df_preprocessed
                st.session_state.scaler = scaler

                st.success("Praproses dan Normalisasi berhasil dilakukan. Data siap untuk klasterisasi!")
                st.subheader("Data Setelah Praproses dan Normalisasi:")
                st.dataframe(st.session_state.df_preprocessed_for_clustering, use_container_width=True, height=300)
                st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True) # Spasi setelah dataframe

elif st.session_state.current_menu == "Klasterisasi Data K-Prototypes":
    st.header("Klasterisasi K-Prototypes")
    if st.session_state.df_preprocessed_for_clustering is None or st.session_state.df_preprocessed_for_clustering.empty:
        st.warning("Silakan lakukan praproses data terlebih dahulu di menu 'Praproses & Normalisasi Data'.")
    else:
        st.markdown("""
        <div style='background-color:#e3f2fd; padding:15px; border-radius:10px; border-left: 5px solid #2196F3;'>
        Pada tahap ini, Anda akan menjalankan algoritma K-Prototypes untuk mengelompokkan siswa.
        <br><br>
        Pilih <b>Jumlah Klaster (K)</b> yang Anda inginkan (antara 2 hingga 6). Algoritma ini akan
        mengelompokkan siswa berdasarkan kombinasi fitur numerik (nilai akademik, kehadiran) dan
        fitur kategorikal (ekstrakurikuler) yang telah disiapkan sebelumnya.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        k = st.slider("Pilih Jumlah Klaster (K)", 2, 6, value=st.session_state.n_clusters, 
                      help="Pilih berapa banyak kelompok siswa yang ingin Anda bentuk.")
        
        if st.button("Jalankan Klasterisasi"):
            with st.spinner(f"Melakukan klasterisasi dengan {k} klaster..."):
                df_clustered, kproto_model, categorical_features_indices = run_kprototypes_clustering(
                    st.session_state.df_preprocessed_for_clustering, k
                )
            
            if df_clustered is not None:
                st.session_state.df_clustered = df_clustered
                st.session_state.kproto_model = kproto_model
                st.session_state.categorical_features_indices = categorical_features_indices
                st.session_state.n_clusters = k
                
                # Merge klaster kembali ke data original untuk tampilan yang lebih informatif
                df_original_with_cluster_display = st.session_state.df_original.copy()
                df_original_with_cluster_display['Klaster'] = df_clustered['Klaster']
                
                st.session_state.cluster_characteristics_map = generate_cluster_descriptions(
                    df_clustered, k, NUMERIC_COLS, CATEGORICAL_COLS
                )

                st.success(f"Klasterisasi selesai dengan {k} klaster! Hasil pengelompokan siswa telah tersedia.")
                
                st.markdown("---") 
                st.subheader("Data Hasil Klasterisasi (Disertai Data Asli):")
                st.dataframe(df_original_with_cluster_display, use_container_width=True, height=300)
                
                st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True) 
                
                st.subheader("Ringkasan Klaster: Jumlah Siswa per Kelompok")
                jumlah_per_klaster = df_original_with_cluster_display["Klaster"].value_counts().sort_index().reset_index()
                jumlah_per_klaster.columns = ["Klaster", "Jumlah Siswa"]
                st.table(jumlah_per_klaster)
                
                st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True) 
            
            if st.session_state.df_clustered is not None:
                st.markdown("---") 
                st.subheader(f"Karakteristik Umum Klaster ({st.session_state.n_clusters} Klaster):")
                st.write("Berikut adalah deskripsi singkat untuk setiap klaster yang terbentuk:")
                
                # Gunakan expander untuk deskripsi klaster agar lebih rapi
                for cluster_id, desc in st.session_state.cluster_characteristics_map.items():
                    with st.expander(f"Klaster {cluster_id}"):
                        st.markdown(desc)
                
                st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True) # Space after expanders

elif st.session_state.current_menu == "Prediksi Klaster Siswa Baru":
    st.header("Prediksi Klaster untuk Siswa Baru")
    if st.session_state.kproto_model is None or st.session_state.scaler is None:
        st.warning("Silakan lakukan klasterisasi terlebih dahulu di menu 'Klasterisasi Data K-Prototypes' untuk melatih model dan scaler.")
    else:
        st.markdown("""
        <div style='background-color:#f1f9ff; padding:15px; border-radius:10px; border-left: 5px solid #2C2F7F;'>
        Halaman ini memungkinkan Anda untuk memprediksi klaster bagi siswa baru. Masukkan data nilai akademik,
        kehadiran, dan keterlibatan ekstrakurikuler siswa. Sistem akan otomatis memproses data
        dan memetakan siswa ke klaster yang paling sesuai berdasarkan model yang telah dilatih.
        <br><br>
        Pemanfaatan klaster membantu guru dalam merancang strategi pembinaan dan pendekatan pembelajaran
        yang lebih personal dan efektif.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---") 
        with st.form("form_input_siswa_baru", clear_on_submit=False): # Non-clear form for easier re-submission
            st.markdown("### Input Data Siswa Baru")
            st.markdown("<div style='margin-bottom: 15px;'></div>", unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Data Akademik & Kehadiran")
                input_rata_nilai = st.number_input("Rata-rata Nilai Akademik (0 - 100)", min_value=0.0, max_value=100.0, value=None, placeholder="Contoh: 85.5", format="%.2f", key="input_nilai_prediksi")
                input_kehadiran = st.number_input("Persentase Kehadiran (0.0 - 1.0)", min_value=0.0, max_value=1.0, value=None, placeholder="Contoh: 0.95 (untuk 95%)", format="%.2f", key="input_kehadiran_prediksi")

            with col2:
                st.markdown("#### Keikutsertaan Ekstrakurikuler")
                st.write("Centang ekstrakurikuler yang diikuti siswa:")
                input_cat_ekskul_values = []
                for idx, col in enumerate(CATEGORICAL_COLS):
                    val = st.checkbox(col.replace("Ekstrakurikuler ", ""), key=f"ekskul_prediksi_{idx}")
                    input_cat_ekskul_values.append(1 if val else 0)
            
            st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
            submitted = st.form_submit_button("Prediksi Klaster Siswa")

        if submitted:
            if input_rata_nilai is None or input_kehadiran is None:
                st.error("Harap isi semua nilai numerik (Rata-rata Nilai Akademik dan Persentase Kehadiran) terlebih dahulu.")
            else:
                input_numeric_data = [input_rata_nilai, input_kehadiran]
                normalized_numeric_data = st.session_state.scaler.transform([input_numeric_data])[0]

                new_student_data_for_prediction = np.array(
                    list(normalized_numeric_data) + input_cat_ekskul_values, dtype=object
                ).reshape(1, -1)

                predicted_cluster = st.session_state.kproto_model.predict(
                    new_student_data_for_prediction, categorical=st.session_state.categorical_features_indices
                )
                
                st.success(f"Prediksi Klaster: Siswa Baru Ini Masuk ke Klaster {predicted_cluster[0]}!")
                st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
                klaster_desc_for_new_student = st.session_state.cluster_characteristics_map.get(predicted_cluster[0], "Deskripsi klaster tidak tersedia.")
                st.markdown(f"""
                <div style='background-color:#e8f5e9; padding:15px; border-radius:10px; border-left: 5px solid #4CAF50;'>
                <b>Karakteristik Klaster {predicted_cluster[0]}:</b><br>
                {klaster_desc_for_new_student}
                <br><br>
                Informasi ini sangat membantu guru dalam memberikan bimbingan dan dukungan yang tepat sasaran
                sesuai dengan profil klaster siswa.
                </div>
                """, unsafe_allow_html=True)

                st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)
                st.subheader("Visualisasi Karakteristik Siswa Baru (Dinormalisasi)")
                st.write("Grafik ini menampilkan nilai fitur siswa setelah dinormalisasi (nilai akademik & kehadiran) atau dalam format biner (ekstrakurikuler).")
                
                values_for_plot = list(normalized_numeric_data) + input_cat_ekskul_values
                labels_for_plot = ["Nilai Akademik (Norm)", "Kehadiran (Norm)"] + [col.replace("Ekstrakurikuler ", "Ekskul\n") for col in CATEGORICAL_COLS]

                fig, ax = plt.subplots(figsize=(10, 6))
                bars = sns.barplot(x=labels_for_plot, y=values_for_plot, palette="viridis", ax=ax)
                
                # Atur batas Y axis agar lebih baik untuk campuran nilai
                ax.set_ylim(min(values_for_plot) - 0.2 if values_for_plot else -1, max(values_for_plot) + 0.2 if values_for_plot else 1)
                
                for index, value in enumerate(values_for_plot):
                    ax.text(bars.patches[index].get_x() + bars.patches[index].get_width() / 2, 
                            bars.patches[index].get_height() + (0.05 if value >= 0 else -0.1), 
                            f"{value:.2f}", ha='center', fontsize=9, weight='bold')

                ax.set_title("Profil Siswa Baru", fontsize=16, weight='bold')
                ax.set_ylabel("Nilai (Dinormalisasi / Biner)")
                plt.xticks(rotation=0) # Pastikan label X tidak miring
                plt.tight_layout()
                st.pyplot(fig)


elif st.session_state.current_menu == "Visualisasi & Profil Klaster":
    st.header("Visualisasi dan Interpretasi Profil Klaster")
    if st.session_state.df_preprocessed_for_clustering is None or st.session_state.df_preprocessed_for_clustering.empty:
        st.warning("Silakan unggah data dan lakukan praproses terlebih dahulu di menu 'Praproses & Normalisasi Data'.")
    else:
        st.markdown("""
        <div style='background-color:#f1f9ff; padding:15px; border-radius:10px; border-left: 5px solid #2C2F7F;'>
        Di halaman ini, Anda dapat memilih jumlah klaster (K) dan melihat visualisasi serta ringkasan
        karakteristik dari setiap kelompok siswa. Visualisasi ini dirancang untuk membantu Anda
        memahami perbedaan utama antara klaster-klaster yang terbentuk.
        <br><br>
        Setiap bar pada grafik merepresentasikan rata-rata (untuk fitur numerik yang dinormalisasi)
        atau modus (untuk fitur kategorikal biner 0/1) dari fitur-fitur di dalam klaster tersebut.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        k_visual = st.slider("Jumlah Klaster (K) untuk visualisasi", 2, 6, value=st.session_state.n_clusters, 
                             help="Geser untuk memilih jumlah klaster yang ingin Anda visualisasikan. Ini akan melatih ulang model sementara untuk tujuan visualisasi.")
        
        # Jalankan klasterisasi ulang hanya untuk tujuan visualisasi jika K berubah
        df_for_visual_clustering, kproto_visual, cat_indices_visual = run_kprototypes_clustering(
            st.session_state.df_preprocessed_for_clustering, k_visual
        )
        
        if df_for_visual_clustering is not None:
            cluster_characteristics_map_visual = generate_cluster_descriptions(
                df_for_visual_clustering, k_visual, NUMERIC_COLS, CATEGORICAL_COLS
            )

            st.markdown(f"### Menampilkan Profil Klaster untuk K = {k_visual}")
            st.write("Visualisasi ini menggunakan data yang telah dinormalisasi (nilai, kehadiran) atau dikodekan (ekstrakurikuler 0/1).")
            
            for i in range(k_visual):
                st.markdown(f"---")
                st.subheader(f"Klaster {i}")
                cluster_data = df_for_visual_clustering[df_for_visual_clustering["Klaster"] == i]

                col1, col2 = st.columns([1, 2])
                with col1:
                    st.markdown("#### Statistik Klaster")
                    st.markdown(f"Jumlah Siswa: {len(cluster_data)}")
                    st.write("Rata-rata Nilai & Kehadiran (Dinormalisasi):")
                    st.dataframe(cluster_data[NUMERIC_COLS].mean().round(2).to_frame(name='Rata-rata'), use_container_width=True)
                    
                    st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)

                    st.write("Kecenderungan Ekstrakurikuler (Modus):")
                    # Tampilkan moda dalam format yang lebih mudah dibaca (0/1 menjadi Ya/Tidak)
                    mode_ekskul_display = cluster_data[CATEGORICAL_COLS].mode().iloc[0].apply(lambda x: 'Ya' if x == '1' else 'Tidak')
                    st.dataframe(mode_ekskul_display.to_frame(name='Paling Umum'), use_container_width=True)

                    st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)
                    st.info(f"Ringkasan Karakteristik Klaster {i}:\n{cluster_characteristics_map_visual.get(i, 'Deskripsi tidak tersedia.')}")

                with col2:
                    st.markdown("#### Grafik Profil Klaster")
                    st.write("📈 Visualisasi ini menunjukkan rata-rata (numerik) atau modus (kategorikal) dari fitur-fitur di klaster ini.")
                    
                    values_for_plot_numeric = cluster_data[NUMERIC_COLS].mean().tolist()
                    values_for_plot_ekskul = [int(cluster_data[col].mode().iloc[0]) for col in CATEGORICAL_COLS]
                    values_for_plot = values_for_plot_numeric + values_for_plot_ekskul

                    labels_for_plot = ["Nilai (Norm)", "Kehadiran (Norm)"] + [col.replace("Ekstrakurikuler ", "Ekskul\n") for col in CATEGORICAL_COLS]

                    fig, ax = plt.subplots(figsize=(10, 6))
                    bars = sns.barplot(x=labels_for_plot, y=values_for_plot, palette="cubehelix", ax=ax)

                    ax.set_ylim(min(values_for_plot) - 0.2 if values_for_plot else -1, max(values_for_plot) + 0.2 if values_for_plot else 1)

                    for index, value in enumerate(values_for_plot):
                        offset = 0.05 if value >= 0 else -0.1
                        ax.text(bars.patches[index].get_x() + bars.patches[index].get_width() / 2, bars.patches[index].get_height() + offset, f"{value:.2f}", ha='center', fontsize=9, weight='bold')

                    ax.set_title(f"Profil Klaster {i}", fontsize=16, weight='bold')
                    ax.set_ylabel("Nilai (Dinormalisasi / Biner)")
                    plt.xticks(rotation=0)
                    plt.tight_layout()
                    st.pyplot(fig)
            
            st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True) # Space after last cluster

elif st.session_state.current_menu == "Lihat Profil Siswa Individual":
    st.header("Lihat Profil Siswa Berdasarkan Nama")
    if st.session_state.df_clustered is None or st.session_state.df_original is None or st.session_state.df_original.empty:
        st.warning("Silakan unggah data di menu 'Unggah Data' dan lakukan klasterisasi di menu 'Klasterisasi Data K-Prototypes' terlebih dahulu.")
    else:
        st.info("Pilih nama siswa dari daftar di bawah untuk melihat detail profil mereka, termasuk klaster tempat mereka berada dan karakteristiknya.")

        st.markdown("---")

        df_original_with_cluster = pd.merge(
            st.session_state.df_original,
            st.session_state.df_clustered[['Klaster']],
            left_index=True, right_index=True,
            how='left'
        )

        # Inisialisasi selected_index untuk memastikan nilai default yang valid
        default_index = 0
        if "selected_student_name" in st.session_state and st.session_state.selected_student_name in df_original_with_cluster["Nama"].unique():
            try:
                default_index = list(df_original_with_cluster["Nama"].unique()).index(st.session_state.selected_student_name)
            except ValueError: # Jika nama tidak ditemukan lagi (misal setelah upload data baru)
                default_index = 0
        
        # Selectbox untuk memilih nama siswa
        nama_terpilih = st.selectbox(
            "Pilih Nama Siswa", 
            df_original_with_cluster["Nama"].unique(), 
            index=default_index, # Mengatur indeks default
            key="pilih_nama_siswa_selectbox", # Menambahkan key unik
            help="Pilih siswa yang profilnya ingin Anda lihat."
        )

        # Simpan nama yang dipilih ke session state agar tetap konsisten saat rerun
        st.session_state.selected_student_name = nama_terpilih
        
        if nama_terpilih:
            siswa_data = df_original_with_cluster[df_original_with_cluster["Nama"] == nama_terpilih].iloc[0]
            klaster_siswa_terpilih = siswa_data['Klaster']
            
            st.success(f"Siswa {nama_terpilih} tergolong dalam Klaster {klaster_siswa_terpilih} (hasil dari {st.session_state.n_clusters} klaster).")
            st.markdown("<div style='margin-top: 20px;'></div>", unsafe_allow_html=True)

            klaster_desc_for_new_student = st.session_state.cluster_characteristics_map.get(klaster_siswa_terpilih, "Deskripsi klaster tidak tersedia.")
            st.markdown(f"""
            <div style='background-color:#f0f4f7; padding:15px; border-radius:10px; border-left: 5px solid {PRIMARY_COLOR};'>
            <b>Karakteristik Klaster Ini:</b><br>
            {klaster_desc_for_new_student}
            </div>
            """, unsafe_allow_html=True)

            st.markdown("---")
            st.subheader("Detail Data Siswa")
            
            col_info, col_chart = st.columns([1, 2]) # Bagi ruang menjadi 1:2
            
            with col_info:
                st.markdown("#### Informasi Dasar")
                st.markdown(f"Nomor Induk: {siswa_data.get('No', '-')}")
                st.markdown(f"Jenis Kelamin: {siswa_data.get('JK', '-')}")
                st.markdown(f"Kelas: {siswa_data.get('Kelas', '-')}")
                st.markdown(f"Rata-rata Nilai Akademik: {siswa_data.get('Rata Rata Nilai Akademik', '-'):.2f}")
                st.markdown(f"Persentase Kehadiran: {siswa_data.get('Kehadiran', '-'):.2%}")

                st.markdown("#### Ekstrakurikuler yang Diikuti")
                ekskul_diikuti_str = []
                for col in CATEGORICAL_COLS:
                    if siswa_data.get(col, 0) == 1:
                        ekskul_diikuti_str.append(col.replace("Ekstrakurikuler ", ""))
                if ekskul_diikuti_str:
                    for ekskul in ekskul_diikuti_str:
                        st.markdown(f"- {ekskul} ✅")
                else:
                    st.markdown("Tidak mengikuti ekstrakurikuler ❌")

            with col_chart:
                st.markdown("#### Visualisasi Profil Siswa Individual")
                st.write("Grafik ini menampilkan nilai asli (tidak dinormalisasi) untuk rata-rata nilai akademik dan persentase kehadiran (0-100%), serta status biner (0/1) untuk ekstrakurikuler.")
                
                labels_siswa_plot = ["Rata-rata\nNilai Akademik", "Kehadiran (%)"] + [col.replace("Ekstrakurikuler ", "Ekskul\n") for col in CATEGORICAL_COLS]

                values_siswa_plot_numeric = [
                    siswa_data["Rata Rata Nilai Akademik"],
                    siswa_data["Kehadiran"] * 100 # Display presence as percentage (0-100)
                ]

                values_siswa_plot_ekskul = [
                    siswa_data[col] * 100 if col in CATEGORICAL_COLS else siswa_data[col] for col in CATEGORICAL_COLS # Tampilkan 0 atau 1 sebagai 0% atau 100% untuk konsistensi skala
                ]

                values_siswa_plot = values_siswa_plot_numeric + values_siswa_plot_ekskul

                fig, ax = plt.subplots(figsize=(10, 6))
                bars = sns.barplot(x=labels_siswa_plot, y=values_siswa_plot, palette="magma", ax=ax)

                # Atur y-limit secara dinamis, pastikan mencakup nilai 0 dan 100
                max_plot_val = max(values_siswa_plot) if values_siswa_plot else 100
                ax.set_ylim(0, max(100, max_plot_val * 1.1)) # Pastikan y-axis tidak terlalu sempit

                for bar, val in zip(bars.patches, values_siswa_plot): # Gunakan patches untuk akses ke bar individual
                    ax.text(bar.get_x() + bar.get_width() / 2, val + (ax.get_ylim()[1] * 0.02), f"{val:.1f}", ha='center', fontsize=9, weight='bold')

                ax.set_title(f"Grafik Profil Siswa - {nama_terpilih}", fontsize=16, weight='bold')
                ax.set_ylabel("Nilai / Status (%)") # Label y-axis lebih informatif
                plt.xticks(rotation=0)
                plt.tight_layout()
                st.pyplot(fig)
            
            st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)
            
            # --- BAGIAN: Menampilkan daftar siswa di klaster yang sama ---
            st.subheader(f"Siswa Lain di Klaster {klaster_siswa_terpilih}:")
            siswa_lain_di_klaster = df_original_with_cluster[
                (df_original_with_cluster['Klaster'] == klaster_siswa_terpilih) & 
                (df_original_with_cluster['Nama'] != nama_terpilih)
            ]
            
            if not siswa_lain_di_klaster.empty:
                st.write("Berikut adalah daftar siswa lain yang juga tergolong dalam klaster ini:")
                # Tampilkan kolom-kolom yang relevan saja untuk daftar ini
                display_cols_for_others = ["No", "Nama", "JK", "Kelas", "Rata Rata Nilai Akademik", "Kehadiran"]
                
                # Format ulang kolom numerik untuk tampilan yang lebih mudah dibaca
                display_df_others = siswa_lain_di_klaster[display_cols_for_others].copy()
                display_df_others["Kehadiran"] = display_df_others["Kehadiran"].apply(lambda x: f"{x:.2%}")

                st.dataframe(display_df_others, use_container_width=True)
            else:
                st.info("Tidak ada siswa lain dalam klaster ini.")
            # --- AKHIR BAGIAN ---
            
            st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)
            st.subheader("Unduh Laporan Profil Siswa (PDF)")
            if st.session_state.cluster_characteristics_map:
                if st.button("Generate & Unduh Laporan PDF", help="Klik untuk membuat laporan PDF profil siswa ini."):
                    with st.spinner("Menyiapkan laporan PDF..."):
                        siswa_data_for_pdf = siswa_data.drop(labels=["Klaster"]).to_dict()
                        # Pastikan semua ID_COLS ada di dict untuk PDF generator
                        for col in ID_COLS:
                            if col not in siswa_data_for_pdf:
                                siswa_data_for_pdf[col] = siswa_data[col]

                        pdf_data_bytes = generate_pdf_profil_siswa(
                            nama_terpilih,
                            siswa_data_for_pdf,
                            siswa_data["Klaster"],
                            st.session_state.cluster_characteristics_map
                        )
                    st.success("Laporan PDF berhasil disiapkan!")
                    st.download_button(
                        label="Klik di Sini untuk Mengunduh PDF",
                        data=pdf_data_bytes,
                        file_name=f"Profil_{nama_terpilih.replace(' ', '_')}.pdf",
                        mime="application/pdf",
                        key="download_profile_pdf_final",
                        help="Klik ini untuk menyimpan laporan PDF ke perangkat Anda."
                    )
            else:
                st.warning("Mohon lakukan klasterisasi terlebih dahulu (Menu 'Klasterisasi Data K-Prototypes') untuk menghasilkan data profil PDF.")
