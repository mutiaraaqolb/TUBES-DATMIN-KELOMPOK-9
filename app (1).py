# =====================================================================
# APLIKASI UTAMA: DASHBOARD PREDIKSI & KLASTERISASI KUALITAS WINE
# =====================================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. KONFIGURASI HALAMAN UTAMA STREAMLIT
st.set_page_config(
    page_title="Red Wine Analytics - Kelompok 9",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Kustomisasi Gaya Tampilan menggunakan CSS internal
st.markdown("""
    <style>
    .main-title { font-size: 38px; font-weight: bold; color: #800020; text-align: center; margin-bottom: 5px; }
    .sub-title { font-size: 16px; text-align: center; color: #555555; margin-bottom: 25px; }
    .section-box { padding: 15px; border-radius: 8px; background-color: #f9f9f9; border-left: 5px solid #800020; margin-bottom: 15px; }
    .metric-card { background-color: #ffffff; padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); text-align: center; }
    </style>
""", unsafe_allow_html=True)


# 2. FUNGSI UNTUK MEMUAT MODEL
@st.cache_resource
def load_models():
    try:
        scaler = joblib.load('scaler.pkl')
        lr_model = joblib.load('lr_model.pkl')
        nb_model = joblib.load('nb_model.pkl')
        kmeans_model = joblib.load('kmeans_model.pkl')

        # feature_cols.pkl bersifat opsional: kalau notebook lama belum
        # menghasilkan file ini, app tetap jalan memakai urutan default
        # di bawah (sama seperti urutan 11 fitur dataset Wine Quality asli).
        try:
            feature_cols = joblib.load('feature_cols.pkl')
        except FileNotFoundError:
            feature_cols = [
                'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
                'chlorides', 'free sulfur dioxide', 'total sulfur dioxide',
                'density', 'pH', 'sulphates', 'alcohol'
            ]

        return scaler, lr_model, nb_model, kmeans_model, feature_cols
    except FileNotFoundError as e:
        st.error(f"⚠️ Gagal memuat file model: {e.filename}. Pastikan file .pkl sudah ditaruh di folder yang sama dengan app.py!")
        return None, None, None, None, None

scaler, lr_model, nb_model, kmeans_model, feature_cols = load_models()


# 3. STRUKTUR SIDEBAR (KONTROL UTAMA INPUT 11 FITUR)
st.sidebar.markdown("<h2 style='color: #800020;'>🍷 Input Parameter Lab</h2>", unsafe_allow_html=True)
st.sidebar.write("Sesuaikan nilai kandungan kimia di bawah ini berdasarkan hasil uji laboratorium:")

# Pembagian slider ke dalam kategori fungsional senyawa kimia agar user-friendly
with st.sidebar.expander("🧪 Parameter Keasaman (Acidity)", expanded=True):
    fixed_acidity = st.slider("Fixed Acidity", 4.0, 16.0, 7.4, 0.1)
    volatile_acidity = st.slider("Volatile Acidity", 0.1, 1.6, 0.7, 0.01)
    citric_acid = st.slider("Citric Acid", 0.0, 1.0, 0.0, 0.01)
    pH = st.slider("Tingkat keasaman (pH)", 2.7, 4.1, 3.51, 0.01)

with st.sidebar.expander("🧂 Parameter Gula & Garam", expanded=True):
    residual_sugar = st.slider("Residual Sugar", 0.9, 15.5, 1.9, 0.1)
    chlorides = st.slider("Chlorides", 0.01, 0.6, 0.076, 0.001)

with st.sidebar.expander("💨 Parameter Sulfur & Densitas", expanded=True):
    free_sulfur_dioxide = st.slider("Free Sulfur Dioxide", 1.0, 72.0, 11.0, 1.0)
    total_sulfur_dioxide = st.slider("Total Sulfur Dioxide", 6.0, 289.0, 34.0, 1.0)
    density = st.slider("Density", 0.990, 1.004, 0.9978, 0.0001)

with st.sidebar.expander("🍺 Parameter Suplemen & Alkohol", expanded=True):
    sulphates = st.slider("Sulphates", 0.3, 2.0, 0.56, 0.01)
    alcohol = st.slider("Alcohol Vol (%)", 8.0, 15.0, 9.4, 0.1)


# 4. Halaman Utama / Main Panel Tampilan Dashboard
st.markdown("<div class='main-title'>Red Wine Quality Analytics Dashboard</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>Sistem Pendukung Keputusan Penjaminan Mutu Produksi Wine Menggunakan Metode CRISP-DM · Kelompok 9</div>", unsafe_allow_html=True)

# Membuat Tabs Navigasi Fitur Utama
tab_prediksi, tab_segmentasi, tab_informasi = st.tabs([
    "🎯 Prediksi Kualitas Supervised", 
    "🧩 Klasterisasi Unsupervised", 
    "📚 Informasi Dataset"
])


# --- TAB 1: PREDIKSI KUALITAS (SUPERVISED) ---
with tab_prediksi:
    st.subheader("Formulir Analisis Prediksi Kualitas")
    st.write("Tab ini menggunakan model klasifikasi terbimbing untuk menentukan status kelayakan komoditas anggur.")
    
    if scaler and lr_model and nb_model:
        # Menyusun inputan slider sidebar ke dalam dictionary agar urutannya
        # bisa disesuaikan otomatis dengan urutan fitur saat training
        # (mengikuti feature_cols.pkl), bukan hardcode urutan manual.
        nilai_input = {
            'fixed acidity': fixed_acidity,
            'volatile acidity': volatile_acidity,
            'citric acid': citric_acid,
            'residual sugar': residual_sugar,
            'chlorides': chlorides,
            'free sulfur dioxide': free_sulfur_dioxide,
            'total sulfur dioxide': total_sulfur_dioxide,
            'density': density,
            'pH': pH,
            'sulphates': sulphates,
            'alcohol': alcohol,
        }
        input_data = np.array([[nilai_input[col] for col in feature_cols]])
        
        # Melakukan transformasi standarisasi data masukan baru
        input_scaled = scaler.transform(input_data)
        
        # Pilihan Model Klasifikasi oleh Pengguna Dashboard
        choice_model = st.selectbox(
            "Pilih Algoritma Klasifikasi Prediksi:",
            ["Logistic Regression", "Gaussian Naïve Bayes (Rekomendasi Akurasi)"]
        )
        
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            if choice_model == "Logistic Regression":
                prediction = lr_model.predict(input_scaled)[0]
                proba = lr_model.predict_proba(input_scaled)[0][prediction] * 100
                st.metric(label="Model Aktif", value="Log Regression")
            else:
                prediction = nb_model.predict(input_scaled)[0]
                proba = nb_model.predict_proba(input_scaled)[0][prediction] * 100
                st.metric(label="Model Aktif", value="Naïve Bayes")
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col2:
            if prediction == 1:
                st.success(f"### 🎉 HASIL ANALISIS: GOOD QUALITY WINE")
                st.write(f"Sistem memprediksi sampel produk anggur merah ini memiliki kualitas **Sangat Baik** dengan tingkat kepastian probabilitas sebesar **{proba:.2f}%**.")
            else:
                st.error(f"### ⚠️ HASIL ANALISIS: BAD QUALITY WINE")
                st.write(f"Sistem memprediksi sampel produk anggur merah ini masuk kategori **Kualitas Rendah** dengan tingkat kepastian probabilitas sebesar **{proba:.2f}%**.")
    else:
        st.warning("Fitur Prediksi belum siap karena file `.pkl` gagal termuat.")


# --- TAB 2: SEGMENTASI VARIETAS (UNSUPERVISED) ---
with tab_segmentasi:
    st.subheader("Segmentasi Karakteristik Kimiawi Wine")
    st.write("Tab ini menggunakan Algoritma K-Means Clustering untuk melihat ke dalam klaster manakah sampel wine saat ini dikelompokkan.")
    
    if scaler and kmeans_model:
        # Menggunakan inputan data berskala yang sama
        cluster_prediction = kmeans_model.predict(input_scaled)[0]
        
        col_c1, col_c2 = st.columns([1, 3])
        
        with col_c1:
            st.markdown(f"""
                <div class='metric-card' style='border-top: 5px solid #008080;'>
                    <h3>Hasil Klaster</h3>
                    <h1 style='color:#008080; font-size:55px;'>{cluster_prediction}</h1>
                    <p style='color:#777;'>Dari Total 3 Klaster (0, 1, 2)</p>
                </div>
            """, unsafe_allow_html=True)
            
        with col_c2:
            st.markdown("<div class='section-box'>", unsafe_allow_html=True)
            st.markdown("#### 🔍 Karakteristik Klaster Terpilih:")
            if cluster_prediction == 0:
                st.write("**Klaster 0 (Standard Blend):** Kelompok wine dengan rata-rata kandungan alkohol sedang, tingkat keasaman stabil, dan karakteristik senyawa yang seimbang untuk pasar massal.")
            elif cluster_prediction == 1:
                st.write("**Klaster 1 (High Acidic Profile):** Kelompok wine yang didominasi oleh konsentrasi volatile acidity dan fixed acidity tinggi, menghasilkan rasa asam yang sangat kuat.")
            else:
                st.write("**Klaster 2 (Premium High-Alcohol):** Kelompok wine yang memiliki karakteristik kadar alkohol dan sulphates yang tinggi, umumnya berkorelasi dengan hasil panen wine berkualitas tinggi.")
            st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.warning("Fitur Klasterisasi belum siap karena file `kmeans_model.pkl` tidak ditemukan.")


# --- TAB 3: INFORMASI DATASET & ATRIBUT ---
with tab_informasi:
    st.subheader("Daftar Definisi Kamus Data (11 Atribut Kimia)")
    
    kamus_data = {
        "Nama Atribut Fitur": [
            "Fixed Acidity", "Volatile Acidity", "Citric Acid", "Residual Sugar", 
            "Chlorides", "Free Sulfur Dioxide", "Total Sulfur Dioxide", 
            "Density", "pH", "Sulphates", "Alcohol"
        ],
        "Deskripsi Fungsi Kimiawi": [
            "Asam utama bawaan dari buah anggur yang tidak mudah menguap (seperti asam tartarat).",
            "Jumlah asam asetat dalam anggur yang jika terlalu tinggi bisa merusak rasa dan memicu rasa cuka.",
            "Ditemukan dalam jumlah kecil, berfungsi menambahkan kesegaran aroma dan rasa buah pada wine.",
            "Sisa kadar gula setelah proses fermentasi selesai, menentukan tingkat kemanisan produk.",
            "Jumlah kandungan garam yang terlarut di dalam cairan wine.",
            "Bentuk gas sulfur bebas yang berfungsi mencegah pertumbuhan mikroba patogen oksidasi.",
            "Total keseluruhan gas sulfur (bebas + terikat), batas konsentrasi pengawet produk.",
            "Tingkat kepekatan massa jenis air anggur, dipengaruhi kadar alkohol dan sisa gula.",
            "Skala keasaman cairan anggur (rentang normal wine biasanya berkisar 3.0 - 4.0 pH).",
            "Bahan aditif suplemen penunjang yang bertindak sebagai antioksidan alami produk.",
            "Persentase volume kadar alkohol murni hasil konversi fermentasi gula buah."
        ]
    }
    st.table(pd.DataFrame(kamus_data))
