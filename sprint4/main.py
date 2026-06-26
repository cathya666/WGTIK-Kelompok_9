"""
NUTRISCAN API - MAIN.PY
Full Azure OpenAI Integration
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import os
import pickle
import re
from dotenv import load_dotenv
import uvicorn
import numpy as np
import logging

# Cek apakah ini proses reloader atau server
if os.environ.get("UVICORN_RELOADER_RUNNING") == "true":
    # Ini proses reloader, skip inisialisasi berat
    pass
else:
    # Ini server, lakukan inisialisasi
    print("✅ Inisialisasi server...")
    # Inisialisasi Azure, model, dll.

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI(title="NutriScan API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================
# AZURE OPENAI CLIENT
# =============================================
AZURE_AVAILABLE = False
azure_client = None

AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-mini")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

def _clean_endpoint(url: str) -> str:
    url = url.strip().rstrip("/")
    url = re.sub(r"/(openai(/v\d+)?|v\d+)$", "", url)
    return url

try:
    from openai import AzureOpenAI
    if AZURE_API_KEY and AZURE_ENDPOINT:
        clean_endpoint = _clean_endpoint(AZURE_ENDPOINT)
        azure_client = AzureOpenAI(
            api_key=AZURE_API_KEY,
            azure_endpoint=clean_endpoint,
            api_version=AZURE_API_VERSION,
        )
        AZURE_AVAILABLE = True
        logger.info(f"✅ Azure OpenAI siap")
        logger.info(f"   Endpoint  : {clean_endpoint}")
        logger.info(f"   Deployment: {AZURE_DEPLOYMENT}")
        logger.info(f"   API Ver   : {AZURE_API_VERSION}")
    else:
        logger.warning("⚠️ .env tidak lengkap")
except Exception as e:
    logger.error(f"⚠️ Gagal inisialisasi Azure OpenAI: {e}")

# =============================================
# SYSTEM PROMPT
# =============================================
SYSTEM_PROMPT = """Anda adalah NutriScan Bot, asisten edukasi kesehatan yang ahli dalam stunting dan gizi anak.

Anda HARUS menjawab dengan INFORMATIF, LENGKAP, dan EMPATIK dalam Bahasa Indonesia.

Aturan:
1. Jawab dengan bahasa Indonesia yang mudah dipahami
2. Gunakan format HTML sederhana untuk keterbacaan (paragraf, bullet points)
3. Berikan jawaban yang panjang dan informatif (minimal 100 kata)
4. Fokus pada edukasi kesehatan anak dan stunting

Topik yang harus Anda kuasai:
- Definisi stunting dan penyebabnya
- Ciri-ciri dan dampak stunting
- Cara mencegah stunting (1000 HPK, ASI, MPASI)
- Makanan bergizi untuk anak
- Kapan harus ke dokter
- Peran orang tua
- Peran ASI dalam pencegahan stunting

Jika pertanyaan di luar topik, arahkan kembali dengan sopan."""

# =============================================
# CHAT HISTORY
# =============================================
chat_history: List[Dict] = []
MAX_HISTORY = 10

def get_ai_response(message: str) -> str:
    global chat_history
    chat_history.append({"role": "user", "content": message})
    
    if len(chat_history) > MAX_HISTORY * 2:
        chat_history = chat_history[-(MAX_HISTORY * 2):]
    
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + chat_history

    try:
        logger.info(f"📤 Sending to Azure: {message[:50]}...")
        
        # Try with max_completion_tokens first
        try:
            response = azure_client.chat.completions.create(
                model=AZURE_DEPLOYMENT,
                messages=messages,
                max_completion_tokens=800,
            )
        except Exception as e:
            logger.warning(f"⚠️ First attempt failed: {e}")
            # Fallback: without parameters
            response = azure_client.chat.completions.create(
                model=AZURE_DEPLOYMENT,
                messages=messages,
            )
        
        reply = (response.choices[0].message.content or "").strip()
        logger.info(f"📥 Response received: {reply[:100] if reply else 'EMPTY'}...")
        
        if not reply:
            logger.warning("⚠️ Empty response, using fallback")
            reply = get_fallback_response(message)
            
        chat_history.append({"role": "assistant", "content": reply})
        return reply
        
    except Exception as e:
        logger.error(f"❌ Azure error: {e}")
        reply = get_fallback_response(message)
        chat_history.append({"role": "assistant", "content": reply})
        return reply

def get_fallback_response(message: str) -> str:
    """Smart fallback responses based on keywords"""
    msg = message.lower().strip()
    
    # Detect question type
    if "asi" in msg and ("peran" in msg or "manfaat" in msg or "apa" in msg or "bagaimana" in msg):
        return get_asi_response()
    elif "stunting" in msg and ("apa" in msg or "definisi" in msg or "itu" in msg):
        return get_stunting_definition()
    elif "penyebab" in msg or "sebab" in msg or "kenapa" in msg:
        return get_causes_response()
    elif "mencegah" in msg or "cegah" in msg or "pencegahan" in msg:
        return get_prevention_response()
    elif "makanan" in msg or "gizi" in msg or "nutrisi" in msg:
        return get_nutrition_response()
    elif "dokter" in msg or "kapan" in msg or "periksa" in msg:
        return get_doctor_response()
    elif "orang tua" in msg or "peran" in msg:
        return get_parents_response()
    elif "ciri" in msg or "tanda" in msg:
        return get_symptoms_response()
    elif "dampak" in msg or "akibat" in msg:
        return get_impact_response()
    else:
        return get_general_response()

def get_asi_response():
    return """<strong>Peran ASI dalam Mencegah Stunting</strong> 🍼

<strong>1. ASI adalah Nutrisi Terbaik untuk Bayi</strong>
ASI mengandung semua nutrisi yang dibutuhkan bayi untuk pertumbuhan optimal dalam 6 bulan pertama kehidupan. Komposisi ASI berubah secara dinamis sesuai dengan kebutuhan bayi yang sedang berkembang.

<strong>2. ASI Eksklusif 6 Bulan</strong>
Berikan ASI saja tanpa makanan atau minuman tambahan selama 6 bulan pertama. ASI eksklusif:
- Memberikan protein, lemak, karbohidrat, vitamin, dan mineral dalam jumlah yang tepat
- Mengandung antibodi (immunoglobulin) yang melindungi bayi dari infeksi
- Mencegah diare dan ISPA yang dapat mengganggu penyerapan gizi
- Mendukung perkembangan otak dan sistem saraf

<strong>3. Kandungan Penting dalam ASI</strong>
- <strong>Kolostrum</strong>: ASI pertama yang kaya antibodi, disebut "vaksin pertama" bayi
- <strong>Laktosa</strong>: Karbohidrat utama untuk energi dan perkembangan otak
- <strong>DHA dan ARA</strong>: Asam lemak esensial untuk perkembangan otak dan penglihatan
- <strong>Protein whey dan kasein</strong>: Mudah dicerna dan tidak membebani ginjal bayi

<strong>4. Manfaat Jangka Panjang ASI</strong>
- Menurunkan risiko stunting hingga 40%
- Meningkatkan daya tahan tubuh
- Mendukung perkembangan kognitif yang lebih baik
- Menurunkan risiko obesitas dan diabetes di masa dewasa

<strong>5. Tips Menyusui yang Berhasil</strong>
- Mulai menyusui dalam 1 jam pertama setelah lahir (Inisiasi Menyusu Dini)
- Susui sesering mungkin (on demand) - minimal 8-12 kali sehari
- Pastikan posisi dan perlekatan yang benar
- Jaga kesehatan ibu dengan makanan bergizi dan istirahat cukup
- Dapatkan dukungan dari keluarga dan tenaga kesehatan

<strong>💡 Pesan Penting:</strong>
ASI adalah investasi terbaik untuk masa depan anak. Kombinasikan ASI eksklusif 6 bulan dengan MPASI bergizi setelahnya untuk pencegahan stunting yang optimal. Konsultasikan dengan bidan atau dokter jika mengalami kesulitan menyusui."""

def get_stunting_definition():
    return """<strong>Apa itu Stunting?</strong> 📚

<strong>Definisi Stunting</strong>
Stunting adalah kondisi gagal tumbuh pada anak balita (di bawah 5 tahun) yang disebabkan oleh kekurangan gizi kronis, sehingga tinggi badan anak berada di bawah standar usianya. Stunting bukan hanya masalah tinggi badan, tetapi juga mempengaruhi perkembangan otak dan kecerdasan anak.

<strong>Ciri-ciri Utama Stunting</strong>
- Tinggi badan di bawah rata-rata anak seusianya
- Pertumbuhan fisik yang lambat atau terhambat
- Wajah tampak lebih muda dari usia sebenarnya
- Perkembangan kognitif dan motorik terlambat

<strong>Penyebab Stunting</strong>
1. <strong>Kurang gizi kronis</strong> - Asupan protein, zat besi, zinc, dan vitamin yang tidak mencukupi
2. <strong>ASI tidak eksklusif</strong> - Diberi makanan/minuman lain sebelum 6 bulan
3. <strong>MPASI tidak tepat</strong> - Kualitas dan kuantitas makanan pendamping ASI yang kurang
4. <strong>Infeksi berulang</strong> - Diare, ISPA, dan infeksi lain yang mengganggu penyerapan gizi
5. <strong>Sanitasi buruk</strong> - Tidak ada akses air bersih dan jamban sehat
6. <strong>Kemiskinan</strong> - Keterbatasan akses ke makanan bergizi dan layanan kesehatan

<strong>Dampak Jangka Panjang</strong>
- Gangguan perkembangan otak dan kecerdasan
- Prestasi belajar yang rendah
- Daya tahan tubuh yang lemah
- Risiko penyakit kronis di masa dewasa
- Produktivitas rendah dan kemiskinan berlanjut

<strong>Pencegahan Stunting</strong>
- Penuhi gizi ibu sejak kehamilan
- Berikan ASI eksklusif 6 bulan
- Berikan MPASI bergizi setelah 6 bulan
- Lengkapi imunisasi dasar
- Jaga kebersihan dan sanitasi
- Pantau pertumbuhan anak di posyandu setiap bulan

Stunting adalah masalah yang dapat dicegah! Deteksi dini dan intervensi tepat sangat penting untuk masa depan anak. Konsultasikan dengan tenaga kesehatan jika Anda memiliki kekhawatiran. 🩺"""

def get_causes_response():
    return """<strong>Penyebab Stunting</strong> 🔍

Stunting disebabkan oleh berbagai faktor yang saling terkait. Berikut penjelasan lengkapnya:

<strong>1. Faktor Gizi (Penyebab Utama)</strong>
- <strong>Kurang Protein</strong>: Protein sangat penting untuk pertumbuhan sel dan jaringan tubuh. Kekurangan protein menyebabkan pertumbuhan terhambat.
- <strong>Defisiensi Zat Besi</strong>: Zat besi diperlukan untuk pembentukan sel darah merah dan perkembangan otak.
- <strong>Kurang Zinc</strong>: Zinc berperan dalam pembelahan sel dan pertumbuhan.
- <strong>Kekurangan Vitamin A</strong>: Penting untuk pertumbuhan tulang dan daya tahan tubuh.
- <strong>ASI Tidak Eksklusif</strong>: Bayi diberikan makanan atau minuman lain sebelum usia 6 bulan.
- <strong>MPASI Tidak Bergizi</strong>: Makanan pendamping ASI yang kurang protein hewani, sayuran, dan buah-buahan.

<strong>2. Faktor Kesehatan</strong>
- <strong>Infeksi Berulang</strong>: Diare, ISPA, dan infeksi lain menyebabkan kehilangan nutrisi.
- <strong>Cacingan</strong>: Mengganggu penyerapan gizi di usus.
- <strong>Berat Badan Lahir Rendah (BBLR)</strong>: Bayi lahir dengan berat kurang dari 2500 gram.
- <strong>Imunisasi Tidak Lengkap</strong>: Meningkatkan risiko penyakit infeksi.

<strong>3. Faktor Lingkungan</strong>
- <strong>Sanitasi Buruk</strong>: Tidak ada akses ke air bersih dan jamban sehat.
- <strong>Kemiskinan</strong>: Keterbatasan ekonomi untuk membeli makanan bergizi.
- <strong>Kurang Pengetahuan</strong>: Orang tua tidak tahu tentang gizi dan pola asuh yang tepat.
- <strong>Stunting pada Ibu</strong>: Ibu hamil yang kekurangan gizi akan melahirkan bayi dengan risiko stunting.

<strong>4. Faktor Sosial</strong>
- <strong>Pendidikan Ibu Rendah</strong>: Mempengaruhi pengetahuan tentang gizi dan kesehatan anak.
- <strong>Akses Kesehatan Terbatas</strong>: Jarak ke fasilitas kesehatan yang jauh.
- <strong>Kurang Stimulasi</strong>: Anak kurang mendapat perhatian dan rangsangan perkembangan.

<strong>💡 Pencegahan Terbaik:</strong>
Mulai dari 1000 hari pertama kehidupan (kehamilan hingga anak berusia 2 tahun). Pastikan ibu hamil mendapat nutrisi cukup, ASI eksklusif 6 bulan, MPASI bergizi, dan imunisasi lengkap."""

def get_prevention_response():
    return """<strong>Cara Mencegah Stunting</strong> 💪

<strong>1. 1000 Hari Pertama Kehidupan (HPK)</strong>
Periode paling kritis dari kehamilan hingga anak berusia 2 tahun. Pastikan:
- Ibu hamil mendapat makanan bergizi (protein, zat besi, asam folat)
- Ibu hamil memeriksakan kehamilan rutin ke bidan/dokter
- Ibu hamil mendapat tablet tambah darah (minimal 90 tablet selama kehamilan)

<strong>2. ASI Eksklusif 6 Bulan</strong>
- Berikan ASI saja tanpa makanan/minuman tambahan selama 6 bulan
- ASI mengandung semua nutrisi dan antibodi yang dibutuhkan bayi
- Lakukan Inisiasi Menyusu Dini (IMD) dalam 1 jam pertama setelah lahir

<strong>3. MPASI Bergizi dan Tepat Waktu</strong>
Mulai usia 6 bulan, berikan makanan pendamping ASI dengan prinsip:
- <strong>Tepat Waktu</strong>: Mulai pada usia 6 bulan, tidak terlalu cepat atau lambat
- <strong>Tepat Kualitas</strong>: Kaya protein hewani (telur, ikan, daging), sayuran, dan buah
- <strong>Tepat Kuantitas</strong>: Jumlah yang cukup sesuai usia anak
- <strong>Bervariasi</strong>: Berbagai jenis makanan untuk nutrisi lengkap
- <strong>Higienis</strong>: Dibuat dengan bahan dan alat yang bersih

<strong>4. Imunisasi Lengkap</strong>
Pastikan anak mendapat imunisasi dasar lengkap:
- BCG (1 kali)
- Polio (4 kali)
- DPT-HB-Hib (3 kali)
- Campak (1 kali)
- Imunisasi lanjutan sesuai jadwal

<strong>5. Sanitasi dan Kebersihan</strong>
- Cuci tangan dengan sabun sebelum makan dan setelah buang air
- Gunakan jamban sehat
- Akses air bersih untuk minum dan memasak
- Jaga kebersihan makanan dan lingkungan

<strong>6. Pantau Pertumbuhan Rutin</strong>
- Bawa anak ke posyandu setiap bulan untuk ditimbang
- Ukur tinggi badan secara teratur
- Catat di Kartu Menuju Sehat (KMS)
- Konsultasikan jika pertumbuhan tidak sesuai

<strong>7. Stimulasi Perkembangan</strong>
- Ajak anak bermain dan berbicara
- Bacakan cerita dan nyanyikan lagu
- Berikan mainan yang merangsang sensorik
- Berikan kasih sayang dan perhatian

<strong>💡 Ingat:</strong> Mencegah lebih baik daripada mengobati. Mulailah sejak dini untuk masa depan anak yang lebih baik! 🌟"""

def get_nutrition_response():
    return """<strong>Makanan Bergizi untuk Anak</strong> 🥗

<strong>1. Protein Hewani (Sangat Penting)</strong>
- <strong>Telur</strong>: Kaya protein dan kolin untuk perkembangan otak
- <strong>Ikan</strong>: Sumber protein dan omega-3 (terutama ikan laut)
- <strong>Daging Ayam</strong>: Protein tinggi dan rendah lemak
- <strong>Daging Sapi</strong>: Kaya zat besi dan protein
- <strong>Hati</strong>: Sumber zat besi dan vitamin A
- <strong>Susu dan Keju</strong>: Sumber protein dan kalsium

<strong>2. Sumber Karbohidrat</strong>
- Nasi, kentang, ubi, singkong
- Roti gandum, pasta, sereal
- Jagung dan biji-bijian

<strong>3. Sayuran Hijau</strong>
- <strong>Bayam</strong>: Kaya zat besi dan vitamin A
- <strong>Brokoli</strong>: Sumber vitamin C dan kalsium
- <strong>Kangkung</strong>: Kaya zat besi dan serat
- <strong>Wortel</strong>: Kaya vitamin A untuk mata

<strong>4. Buah-buahan</strong>
- <strong>Pisang</strong>: Sumber energi dan kalium
- <strong>Pepaya</strong>: Kaya vitamin C dan serat
- <strong>Jeruk</strong>: Sumber vitamin C untuk daya tahan tubuh
- <strong>Alpukat</strong>: Lemak sehat untuk perkembangan otak

<strong>5. Contoh Menu MPASI Sehat</strong>
- <strong>Usia 6-8 bulan</strong>: Bubur nasi halus + ikan + sayuran cincang halus
- <strong>Usia 9-11 bulan</strong>: Nasi tim + telur + wortel + bayam
- <strong>Usia 12-23 bulan</strong>: Nasi + lauk (ayam/ikan/telur) + sayur + buah
- <strong>Camilan Sehat</strong>: Pisang, pepaya, biskuit bayi, puding susu

<strong>6. Tips Memberikan Makanan</strong>
- Berikan makanan dalam porsi kecil tapi sering (3 kali makan utama + 2 kali camilan)
- Variasikan jenis makanan untuk nutrisi lengkap
- Buat makanan dengan tampilan menarik
- Ajak anak makan bersama keluarga
- Jangan memaksa anak makan, tetapi tawarkan dengan sabar

<strong>💡 Ingat:</strong> Pola makan yang baik dimulai sejak dini. Jadikan makan sebagai momen menyenangkan! 🍽️"""

def get_doctor_response():
    return """<strong>Kapan Harus ke Dokter?</strong> 🏥

<strong>1. Masalah Pertumbuhan</strong>
Segera bawa anak ke dokter atau puskesmas jika:
- Berat badan tidak naik dalam 2 bulan berturut-turut
- Tinggi badan tidak bertambah sesuai usia
- Berat badan turun drastis dalam waktu singkat
- Pertumbuhan terlihat sangat lambat dibanding anak seusianya

<strong>2. Masalah Makan</strong>
- Anak menolak makan terus-menerus (lebih dari 3 hari)
- Muntah yang berulang atau berkepanjangan
- Diare lebih dari 3 kali sehari selama lebih dari 2 hari
- Nafsu makan sangat buruk dan tidak membaik

<strong>3. Tanda-tanda Gizi Buruk</strong>
- Badan sangat kurus atau terlihat tulang iga jelas
- Pembengkakan pada kaki dan tangan (edema)
- Rambut tipis, kering, dan mudah rontok
- Kulit kering, bersisik, atau berubah warna
- Anak terlihat lemah, lesu, dan tidak aktif

<strong>4. Sering Sakit</strong>
- Demam tinggi (di atas 38.5°C) yang tidak turun dalam 24 jam
- Batuk pilek yang tidak sembuh dalam 2 minggu
- Diare berdarah atau muntah terus-menerus
- Kejang atau kesadaran menurun

<strong>5. Pemeriksaan Rutin</strong>
- Bawa anak ke posyandu setiap bulan untuk penimbangan dan pengukuran
- Lakukan pemeriksaan kesehatan rutin setiap 3-6 bulan
- Pastikan imunisasi lengkap sesuai jadwal
- Periksa ke dokter gigi setiap 6 bulan

<strong>6. Saat Ada Kekhawatiran</strong>
Jika Anda merasa ada yang tidak beres dengan pertumbuhan atau perkembangan anak:
- Jangan ragu untuk bertanya ke tenaga kesehatan
- Catat pertumbuhan dan perkembangan anak
- Bawa catatan tersebut saat ke dokter

<strong>💡 Pesan Penting:</strong>
Deteksi dini sangat penting untuk penanganan yang tepat. Jangan menunda jika Anda memiliki kekhawatiran. Konsultasikan dengan tenaga kesehatan terdekat! 🩺"""

def get_parents_response():
    return """<strong>Peran Orang Tua dalam Mencegah Stunting</strong> 👨‍👩‍👧

<strong>1. Pemenuhan Gizi yang Tepat</strong>
- Pastikan ibu hamil mendapat makanan bergizi (protein, zat besi, asam folat)
- Berikan ASI eksklusif 6 bulan pertama kehidupan
- Berikan MPASI bergizi dan bervariasi setelah 6 bulan
- Sediakan makanan sehat untuk seluruh keluarga
- Batasi makanan tinggi gula, garam, dan lemak jenuh

<strong>2. Pemantauan Pertumbuhan</strong>
- Bawa anak ke posyandu setiap bulan untuk ditimbang dan diukur
- Isi dan pahami Kartu Menuju Sehat (KMS)
- Ukur tinggi badan anak secara teratur
- Catat perkembangan anak di buku KIA
- Konsultasikan jika pertumbuhan tidak sesuai

<strong>3. Stimulasi Perkembangan</strong>
- Ajak anak bermain dan berbicara setiap hari
- Bacakan cerita dan nyanyikan lagu untuk merangsang bahasa
- Berikan mainan yang merangsang sensorik dan motorik
- Latih anak untuk duduk, merangkak, berjalan sesuai usia
- Berikan kasih sayang dan perhatian yang cukup

<strong>4. Kebersihan dan Kesehatan</strong>
- Terapkan Perilaku Hidup Bersih dan Sehat (PHBS)
- Cuci tangan dengan sabun sebelum makan dan setelah buang air
- Jaga kebersihan makanan yang akan dikonsumsi
- Pastikan akses air bersih untuk minum dan memasak
- Lengkapi imunisasi dasar anak

<strong>5. Edukasi dan Pengetahuan</strong>
- Belajar tentang gizi dan kesehatan anak
- Ikut serta dalam kegiatan posyandu dan penyuluhan
- Bertanya pada tenaga kesehatan jika ada keraguan
- Bergabung dengan komunitas orang tua untuk berbagi pengalaman
- Mengikuti perkembangan informasi kesehatan anak

<strong>6. Dukungan Psikologis</strong>
- Berikan perhatian dan kasih sayang yang cukup
- Ciptakan lingkungan yang aman dan nyaman untuk anak
- Hindari stres pada ibu karena dapat mempengaruhi ASI
- Dukung pasangan dalam merawat anak
- Berbagi beban dan tanggung jawab dalam keluarga

<strong>💡 Ingat:</strong>
Orang tua adalah kunci utama dalam mencegah stunting. Mulailah dari hal kecil dan lakukan dengan konsisten. Investasi terbaik adalah untuk masa depan anak! 🌟"""

def get_symptoms_response():
    return """<strong>Ciri-Ciri Anak Stunting</strong> 🔍

<strong>Ciri Fisik yang Terlihat:</strong>
1. <strong>Tinggi Badan di Bawah Rata-rata</strong>
   - Tinggi anak jauh di bawah standar usianya
   - Pertumbuhan tinggi badan sangat lambat
   - Terlihat lebih pendek dari anak seusianya

2. <strong>Berat Badan Kurang</strong>
   - Berat badan tidak naik sesuai usia
   - Badan terlihat kurus atau kecil
   - Lingkar lengan atas (LILA) kecil

3. <strong>Wajah Tampak Lebih Muda</strong>
   - Fitur wajah seperti bayi (baby face)
   - Tidak sesuai dengan usia kronologis

4. <strong>Perkembangan Fisik Terlambat</strong>
   - Gigi tumbuh terlambat
   - Ubun-ubun besar lambat menutup
   - Tulang tidak berkembang optimal

<strong>Ciri Perilaku dan Perkembangan:</strong>
1. <strong>Motorik Terlambat</strong>
   - Telat duduk (seharusnya 6-8 bulan)
   - Telat merangkak (seharusnya 8-10 bulan)
   - Telat berjalan (seharusnya 12-15 bulan)

2. <strong>Kognitif Terhambat</strong>
   - Kurang responsif terhadap rangsangan
   - Perkembangan bahasa terlambat
   - Sulit berkonsentrasi dan belajar
   - Kemampuan berpikir lambat

3. <strong>Perilaku</strong>
   - Anak terlihat lesu dan kurang aktif
   - Sering rewel atau menangis tanpa sebab
   - Kurang antusias saat bermain
   - Interaksi sosial terbatas

<strong>Ciri Kesehatan:</strong>
1. <strong>Sering Sakit</strong>
   - Infeksi saluran pernapasan akut (ISPA) berulang
   - Diare yang sering terjadi
   - Demam berulang tanpa sebab jelas

2. <strong>Daya Tahan Tubuh Rendah</strong>
   - Mudah terkena infeksi
   - Penyembuhan yang lambat
   - Kondisi kesehatan umum yang buruk

3. <strong>Masalah Pencernaan</strong>
   - Penyerapan nutrisi terganggu
   - Sering mengalami diare
   - Nafsu makan buruk

<strong>💡 Penting untuk Diingat:</strong>
Jika Anda melihat satu atau lebih ciri-ciri di atas, segera konsultasikan dengan dokter atau tenaga kesehatan. Deteksi dini dan penanganan cepat sangat penting untuk mencegah dampak jangka panjang. Jangan menunda untuk memeriksakan anak Anda! 🩺"""

def get_impact_response():
    return """<strong>Dampak Jangka Panjang Stunting</strong> ⚠️

<strong>1. Dampak pada Perkembangan Otak</strong>
🧠 <strong>Gangguan Kognitif:</strong>
- Perkembangan otak tidak optimal
- Kemampuan belajar dan berpikir menurun
- IQ lebih rendah dari rata-rata
- Sulit berkonsentrasi dan fokus di sekolah
- Gangguan memori jangka pendek

<strong>2. Dampak pada Kesehatan Fisik</strong>
- <strong>Daya Tahan Tubuh Lemah</strong>: Mudah terkena infeksi dan penyakit
- <strong>Penyakit Kronis</strong>: Risiko tinggi terkena diabetes, hipertensi, dan penyakit jantung di masa dewasa
- <strong>Postur Tubuh Tidak Proporsional</strong>: Tubuh pendek dengan kepala yang relatif besar
- <strong>Produktivitas Rendah</strong>: Kapasitas fisik untuk bekerja berkurang

<strong>3. Dampak pada Pendidikan</strong>
📚 <strong>Prestasi Belajar Menurun:</strong>
- Nilai sekolah di bawah rata-rata
- Sulit mengikuti pelajaran di kelas
- Risiko putus sekolah lebih tinggi
- Kemampuan membaca dan berhitung terhambat

<strong>4. Dampak Sosial dan Emosional</strong>
- <strong>Kurang Percaya Diri</strong>: Merasa berbeda dari teman-temannya
- <strong>Interaksi Sosial Terbatas</strong>: Sulit bergaul dan berteman
- <strong>Risiko Depresi</strong>: Lebih rentan terhadap stres dan kecemasan
- <strong>Keterasingan</strong>: Dijauhi atau dibully oleh teman sebaya

<strong>5. Dampak Ekonomi</strong>
💰 <strong>Produktivitas Rendah:</strong>
- Pendapatan lebih rendah saat dewasa (sekitar 20-30% lebih rendah)
- Sulit bersaing di dunia kerja
- Kemiskinan berlanjut ke generasi berikutnya
- Beban ekonomi bagi keluarga dan negara

<strong>6. Dampak pada Generasi Berikutnya</strong>
🔄 <strong>Siklus Stunting Berlanjut:</strong>
- Perempuan yang stunting akan melahirkan bayi dengan berat rendah
- Bayi dengan berat rendah berisiko stunting
- Stunting menjadi masalah antar generasi
- Siklus kemiskinan dan malnutrisi sulit diputus

<strong>💡 Pesan Penting:</strong>
Stunting bukan hanya masalah tinggi badan, tetapi mempengaruhi seluruh aspek kehidupan anak. Pencegahan adalah investasi terbaik untuk masa depan anak dan generasi mendatang. Mulailah dari 1000 hari pertama kehidupan! 🌟"""

def get_general_response():
    return """<strong>Tentang Stunting</strong> 📚

Stunting adalah kondisi gagal tumbuh pada anak balita yang disebabkan oleh kekurangan gizi kronis, sehingga tinggi badan anak berada di bawah standar usianya.

<strong>Poin Penting yang Perlu Diketahui:</strong>
1. Stunting dapat dicegah dengan gizi yang cukup sejak 1000 hari pertama kehidupan
2. ASI eksklusif 6 bulan dan MPASI bergizi adalah kunci pencegahan
3. Pantau pertumbuhan anak secara rutin di posyandu setiap bulan
4. Konsultasikan dengan dokter jika ada kekhawatiran tentang pertumbuhan anak
5. Stunting berdampak jangka panjang pada kecerdasan dan produktivitas

<strong>Topik yang Bisa Saya Bantu:</strong>
• Apa itu stunting? - Definisi dan penjelasan lengkap
• Penyebab stunting - Faktor-faktor yang menyebabkan
• Cara mencegah stunting - Strategi pencegahan efektif
• Makanan bergizi untuk anak - Rekomendasi menu sehat
• Kapan harus ke dokter? - Tanda-tanda yang perlu diwaspadai
• Peran orang tua - Apa yang bisa dilakukan orang tua
• Ciri-ciri stunting - Tanda-tanda yang perlu diperhatikan
• Dampak stunting - Akibat jangka panjang
• Peran ASI - Manfaat ASI untuk pencegahan stunting

Silakan ketik pertanyaan Anda atau klik topik di atas! Saya siap membantu Anda. 😊"""

# =============================================
# LOAD MODEL
# =============================================
ml_model = None
model_loaded = False
scaler = None

try:
    if os.path.exists("model_stunting.pkl"):
        with open("model_stunting.pkl", "rb") as f:
            ml_model = pickle.load(f)
        model_loaded = True
        logger.info("✅ Model ML dimuat")
    else:
        logger.warning("⚠️ model_stunting.pkl tidak ditemukan")
    
    if os.path.exists("scaler.pkl"):
        with open("scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        logger.info("✅ Scaler dimuat")
except Exception as e:
    logger.error(f"⚠️ Gagal load model: {e}")

# =============================================
# PYDANTIC MODELS
# =============================================
class PredictRequest(BaseModel):
    age: float
    gender: str
    height: float
    weight: float

class PredictResponse(BaseModel):
    status: str
    risk_score: float
    bmi: float
    message: str
    details: Optional[Dict[str, Any]] = None

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    azure_used: bool

# =============================================
# ENDPOINTS
# =============================================
@app.get("/")
async def root():
    return FileResponse("stunting_app.html")

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "azure_available": AZURE_AVAILABLE,
        "azure_deployment": AZURE_DEPLOYMENT,
        "model_loaded": model_loaded,
    }

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    message = (request.message or "").strip()
    if not message:
        return ChatResponse(response="Silakan ketik pertanyaan Anda.", azure_used=False)
    
    logger.info(f"📨 Chat request: {message[:50]}...")
    
    # Always use Azure first if available
    if AZURE_AVAILABLE and azure_client:
        try:
            reply = get_ai_response(message)
            # Clean up any markdown or ** symbols
            reply = reply.replace('**', '').replace('__', '')
            return ChatResponse(response=reply, azure_used=True)
        except Exception as e:
            logger.error(f"❌ Chat error: {e}")
            reply = get_fallback_response(message)
            reply = reply.replace('**', '').replace('__', '')
            return ChatResponse(response=reply, azure_used=False)
    else:
        reply = get_fallback_response(message)
        reply = reply.replace('**', '').replace('__', '')
        return ChatResponse(response=reply, azure_used=False)

@app.post("/chat/reset")
async def chat_reset():
    global chat_history
    chat_history = []
    return {"status": "ok"}

def calculate_z_scores(age, gender, height, weight):
    """Calculate WHO Z-scores for height and weight"""
    WHO_HEIGHT_DATA = {
        'Laki-laki': {
            0: 49.9, 1: 53.2, 2: 56.4, 3: 59.4, 4: 62.0,
            5: 64.4, 6: 66.6, 7: 68.7, 8: 70.6, 9: 72.5,
            10: 74.3, 11: 76.0, 12: 77.7, 13: 79.3, 14: 80.9,
            15: 82.3, 16: 83.8, 17: 85.2, 18: 86.5, 19: 87.8,
            20: 89.1, 21: 90.3, 22: 91.4, 23: 92.5, 24: 93.6,
            30: 97.0, 36: 100.5, 42: 103.6, 48: 106.5, 54: 109.2, 60: 111.8
        },
        'Perempuan': {
            0: 49.1, 1: 52.4, 2: 55.5, 3: 58.2, 4: 60.6,
            5: 63.0, 6: 65.2, 7: 67.3, 8: 69.2, 9: 71.0,
            10: 72.7, 11: 74.4, 12: 76.0, 13: 77.5, 14: 79.0,
            15: 80.5, 16: 81.9, 17: 83.3, 18: 84.6, 19: 85.9,
            20: 87.1, 21: 88.3, 22: 89.4, 23: 90.5, 24: 91.6,
            30: 95.5, 36: 99.1, 42: 102.4, 48: 105.5, 54: 108.2, 60: 110.4
        }
    }
    
    WHO_WEIGHT_DATA = {
        'Laki-laki': {
            0: 3.3, 1: 4.3, 2: 5.2, 3: 6.0, 4: 6.7,
            5: 7.3, 6: 7.9, 7: 8.4, 8: 8.9, 9: 9.4,
            10: 9.8, 11: 10.2, 12: 10.6, 13: 11.0, 14: 11.4,
            15: 11.8, 16: 12.1, 17: 12.4, 18: 12.7, 19: 13.0,
            20: 13.3, 21: 13.6, 22: 13.9, 23: 14.2, 24: 14.5,
            30: 16.0, 36: 17.3, 42: 18.5, 48: 19.6, 54: 20.6, 60: 21.5
        },
        'Perempuan': {
            0: 3.2, 1: 4.2, 2: 5.0, 3: 5.8, 4: 6.4,
            5: 6.9, 6: 7.3, 7: 7.8, 8: 8.2, 9: 8.6,
            10: 9.0, 11: 9.4, 12: 9.8, 13: 10.2, 14: 10.6,
            15: 10.9, 16: 11.2, 17: 11.5, 18: 11.8, 19: 12.1,
            20: 12.4, 21: 12.7, 22: 13.0, 23: 13.3, 24: 13.6,
            30: 15.2, 36: 16.7, 42: 18.0, 48: 19.2, 54: 20.2, 60: 21.2
        }
    }
    
    def interpolate(data_dict, age):
        ages = sorted(data_dict.keys())
        if age <= ages[0]:
            return data_dict[ages[0]]
        if age >= ages[-1]:
            return data_dict[ages[-1]]
        for i in range(len(ages)-1):
            if ages[i] <= age <= ages[i+1]:
                ratio = (age - ages[i]) / (ages[i+1] - ages[i])
                return data_dict[ages[i]] + ratio * (data_dict[ages[i+1]] - data_dict[ages[i]])
        return data_dict[ages[-1]]
    
    height_median = interpolate(WHO_HEIGHT_DATA[gender], age)
    weight_median = interpolate(WHO_WEIGHT_DATA[gender], age)
    
    height_sd = 3.5 + (age / 60) * 2.5
    weight_sd = weight_median * 0.15
    
    height_z = (height - height_median) / height_sd
    weight_z = (weight - weight_median) / weight_sd
    
    return height_z, weight_z, height_median, weight_median

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    # Validation
    if request.gender not in ["Laki-laki", "Perempuan"]:
        raise HTTPException(status_code=400, detail="Gender tidak valid. Gunakan 'Laki-laki' atau 'Perempuan'")
    
    if request.age < 0 or request.age > 60:
        raise HTTPException(status_code=400, detail="Usia harus 0-60 bulan")
    
    if request.height < 40 or request.height > 120:
        raise HTTPException(status_code=400, detail="Tinggi badan tidak valid (40-120 cm)")
    
    if request.weight < 2 or request.weight > 25:
        raise HTTPException(status_code=400, detail="Berat badan tidak valid (2-25 kg)")
    
    bmi = round(request.weight / ((request.height / 100) ** 2), 2)
    
    height_z, weight_z, height_median, weight_median = calculate_z_scores(
        request.age, request.gender, request.height, request.weight
    )
    
    # Stunting detection
    is_stunted = height_z < -2.0
    is_severe_stunted = height_z < -3.0
    is_wasted = weight_z < -2.0
    is_severe_wasted = weight_z < -3.0
    
    if is_severe_stunted and is_severe_wasted:
        status = "Stunting Berat + Gizi Buruk Berat"
        message = "⚠️ PERINGATAN: Anak mengalami stunting berat DAN gizi buruk berat. SEGERA bawa ke dokter atau puskesmas terdekat!"
        risk_score = 0.95
    elif is_severe_stunted:
        status = "Stunting Berat"
        message = "⚠️ Anak mengalami stunting berat. Segera konsultasi ke dokter atau puskesmas!"
        risk_score = 0.85
    elif is_stunted and is_wasted:
        status = "Stunting + Gizi Buruk"
        message = "⚠️ Anak mengalami stunting dan gizi buruk. Segera konsultasi ke dokter!"
        risk_score = 0.80
    elif is_stunted:
        status = "Stunting"
        message = "⚠️ Anak mengalami stunting. Segera konsultasi ke dokter/puskesmas untuk penanganan lebih lanjut."
        risk_score = 0.70
    elif is_wasted:
        status = "Gizi Buruk (Wasting)"
        message = "⚠️ Berat badan di bawah normal (gizi buruk). Segera bawa ke dokter/puskesmas!"
        risk_score = 0.75
    else:
        if height_z < -1.5:
            status = "Normal (Perlu Monitoring)"
            message = "✅ Status normal, tetapi tinggi badan mendekati batas bawah. Perhatikan asupan gizi dan pantau pertumbuhan."
            risk_score = 0.25
        elif height_z < -1.0:
            status = "Normal (Perhatikan Gizi)"
            message = "✅ Status normal. Pertahankan pola makan bergizi dan pantau pertumbuhan secara teratur."
            risk_score = 0.10
        else:
            status = "Normal"
            message = "✅ Status normal. Pertahankan pola makan bergizi dan stimulasi tumbuh kembang yang baik."
            risk_score = 0.05
    
    # ML Model
    if model_loaded and ml_model:
        try:
            gender_encoded = 1 if request.gender == "Laki-laki" else 0
            input_data = np.array([[request.age, gender_encoded, request.height, request.weight]])
            
            if scaler is not None:
                input_data = scaler.transform(input_data)
            
            if hasattr(ml_model, 'predict_proba'):
                prob = ml_model.predict_proba(input_data)[0]
                ml_risk = float(prob[1]) if len(prob) > 1 else 0.5
                risk_score = (risk_score * 0.6 + ml_risk * 0.4)
        except Exception as e:
            logger.error(f"⚠️ Model prediction error: {e}")
    
    risk_score = min(max(risk_score, 0.01), 0.99)
    
    return PredictResponse(
        status=status,
        risk_score=round(risk_score, 3),
        bmi=bmi,
        message=message,
        details={
            "age": request.age,
            "gender": request.gender,
            "height": request.height,
            "weight": request.weight,
            "height_z_score": round(height_z, 2),
            "weight_z_score": round(weight_z, 2),
            "height_median": round(height_median, 1),
            "weight_median": round(weight_median, 1),
            "bmi": bmi,
        }
    )

if __name__ == "__main__":
    # Cek apakah ini proses anak atau proses reloader
    if not os.environ.get("UVICORN_RELOADER_RUNNING"):
        print("=" * 55)
        print("🚀 NUTRISCAN API v2.0")
        print("=" * 55)
        print(f"📊 Model ML: {'✅ Tersedia' if model_loaded else '❌ Tidak tersedia'}")
        print(f"🤖 Azure AI: {'✅ Tersedia' if AZURE_AVAILABLE else '❌ Tidak tersedia'}")
        print("=" * 55)
    
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)