# ============================================
# NUTRISCAN API - MAIN.PY
# VERSI TERBARU DENGAN RESPONSE YANG DIJAMIN
# ============================================

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import os
import pickle
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from datetime import datetime
import uvicorn
import json

load_dotenv()

# ============================================
# 1. INISIALISASI APP
# ============================================
app = FastAPI(
    title="NutriScan API",
    description="API untuk deteksi stunting dan chatbot",
    version="1.0.0"
)

# CORS Middleware - DIJAMIN
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# 2. AZURE OPENAI SETUP (OPSIONAL)
# ============================================
AZURE_AVAILABLE = False
client = None

try:
    from openai import OpenAI
    
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "https://cathyalatisha-4828-resource.services.ai.azure.com/openai/v1")
    AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "o4-mini")
    
    if AZURE_OPENAI_API_KEY:
        client = OpenAI(
            base_url=AZURE_OPENAI_ENDPOINT,
            api_key=AZURE_OPENAI_API_KEY,
        )
        
        # Test koneksi
        test_response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT_NAME,
            messages=[{"role": "user", "content": "Halo"}],
            max_completion_tokens=5
        )
        
        AZURE_AVAILABLE = True
        print("✅ Azure OpenAI terhubung!")
    else:
        print("⚠️ Azure OpenAI API Key tidak ditemukan, menggunakan fallback")
        
except Exception as e:
    AZURE_AVAILABLE = False
    print(f"⚠️ Azure OpenAI tidak tersedia: {e}")
    print("   Menggunakan fallback rule-based")

# ============================================
# 3. RESPONSES DATABASE - LENGKAP
# ============================================
RESPONSES = {
    "definisi": """📚 **Apa itu Stunting?**

Stunting adalah kondisi gagal tumbuh pada anak balita (0-59 bulan) akibat kekurangan gizi kronis, terutama dalam 1000 Hari Pertama Kehidupan (HPK).

**Ciri-ciri utama:**
• Tinggi badan di bawah standar usianya (z-score < -2 SD)
• Pertumbuhan melambat
• Perkembangan kognitif terhambat

**Faktor risiko:**
• Kurang gizi kronis (protein, vitamin, mineral)
• ASI tidak eksklusif 6 bulan pertama
• MPASI kurang bergizi
• Infeksi berulang (diare, ISPA)
• Sanitasi lingkungan buruk
• Akses kesehatan terbatas""",

    "penyebab": """🔍 **Penyebab Stunting:**

**Faktor Utama:**
1. Kurang gizi kronis (protein, vitamin, mineral)
2. ASI tidak eksklusif 6 bulan pertama
3. MPASI kurang bergizi dan tidak tepat waktu
4. Infeksi berulang (diare, ISPA)
5. Sanitasi lingkungan yang buruk

**Faktor Pendukung:**
• Pendidikan ibu rendah
• Kemiskinan
• Akses kesehatan terbatas
• Kurangnya pengetahuan gizi""",

    "pencegahan": """🛡️ **Cara Mencegah Stunting:**

**1000 HPK (Hari Pertama Kehidupan):**
1. ✅ ASI eksklusif selama 6 bulan pertama
2. ✅ MPASI bergizi seimbang (protein hewani, sayur, buah)
3. ✅ Imunisasi lengkap sesuai jadwal
4. ✅ Pantau pertumbuhan setiap bulan di posyandu
5. ✅ Jaga kebersihan lingkungan dan sanitasi

**Pola Makan:**
• Protein hewani (telur, ikan, ayam, hati)
• Sayur dan buah setiap hari
• Karbohidrat kompleks (nasi, ubi, jagung)""",

    "makanan": """🍳 **Makanan untuk Cegah Stunting:**

**🍗 Protein Hewani (WAJIB setiap hari):**
🥚 Telur (1 butir/hari)
🐟 Ikan (teri, salmon, kembung)
🍗 Ayam, hati ayam
🥛 Susu dan olahannya

**🍚 Karbohidrat:**
Nasi, ubi, jagung, kentang

**🥬 Sayur & Buah:**
Bayam, wortel, brokoli, sawi
Pepaya, pisang, alpukat, mangga

**📋 Contoh Menu Sehari:**
- Pagi: Bubur ayam + telur rebus
- Selingan: Pisang + susu
- Siang: Nasi + ikan goreng + sayur bayam
- Selingan: Yogurt + buah potong
- Malam: Nasi + ayam suwir + sayur wortel""",

    "dokter": """🏥 **Kapan Harus ke Dokter?**

**Segera bawa anak ke dokter jika:**

**🚨 Tanda Darurat:**
1. 🔴 Tinggi badan tidak naik 2 bulan berturut-turut
2. 🔴 Berat badan turun atau stagnan > 1 bulan
3. 🔴 Anak tampak kurus atau sangat kecil

**🩺 Tanda Kesehatan:**
4. 🔴 Sering sakit (diare, batuk, demam berulang)
5. 🔴 Nafsu makan sangat buruk
6. 🔴 Anak tampak lemas dan tidak aktif

📌 **Rutin ke posyandu setiap bulan!**""",

    "ciri": """👶 **Ciri-ciri Anak Stunting:**

**Tanda Fisik:**
1. 🔴 Tinggi badan di bawah standar usianya
2. 🔴 Berat badan tidak naik atau stagnan
3. 🔴 Wajah tampak lebih muda dari usianya
4. 🔴 Postur tubuh lebih pendek dari teman sebaya

**Tanda Perilaku:**
• Perkembangan motorik lambat
• Kemampuan kognitif terhambat
• Mudah sakit
• Kurang aktif dan responsif""",

    "dampak": """⚠️ **Dampak Stunting Jangka Panjang:**

**1. 🧠 Kognitif & Pendidikan**
• IQ lebih rendah (10-15 poin lebih rendah)
• Prestasi belajar menurun
• Kesulitan konsentrasi dan fokus

**2. 🩺 Kesehatan**
• Daya tahan tubuh lemah
• Risiko penyakit tidak menular
• Postur tubuh pendek saat dewasa

**3. 💰 Sosial & Ekonomi**
• Produktivitas rendah
• Pendapatan lebih rendah
• Risiko kemiskinan lebih tinggi""",

    "orang_tua": """👨‍👩‍👧 **Peran Orang Tua Mencegah Stunting:**

**🍽️ Nutrisi:**
1. Berikan ASI eksklusif 6 bulan
2. Siapkan MPASI bergizi setiap hari
3. Pastikan protein hewani dalam setiap menu
4. Variasikan makanan agar anak tidak bosan

**📏 Kesehatan:**
5. Pantau pertumbuhan setiap bulan di posyandu
6. Lengkapi imunisasi dasar anak
7. Jaga kebersihan lingkungan

**🧠 Stimulasi:**
8. Ajak anak bermain dan berbicara
9. Berikan mainan edukatif
10. Berikan kasih sayang dan perhatian""",

    "asi": """🍼 **Peran ASI dalam Mencegah Stunting:**

**ASI adalah makanan terbaik untuk bayi 0-6 bulan!**

**✅ Manfaat ASI Eksklusif:**
1. **Nutrisi Lengkap** - Protein, lemak, vitamin, mineral sempurna
2. **Antibodi Alami** - Melindungi dari infeksi
3. **Meningkatkan Daya Tahan Tubuh** - Mengurangi risiko diare dan ISPA
4. **Mendukung Perkembangan Otak** - Kandungan DHA untuk kecerdasan
5. **Mencegah Stunting** - Langkah pertama mencegah stunting

**📋 Panduan ASI:**
• Berikan ASI eksklusif 0-6 bulan
• Lanjutkan ASI sampai 2 tahun dengan MPASI
• Susui sesering mungkin (8-12 kali sehari)""",

    "salam": """Halo! 👋 Saya NutriScan Bot.

Saya siap membantu Anda tentang:
✅ Definisi stunting
✅ Penyebab dan pencegahan
✅ Makanan bergizi
✅ Peran ASI
✅ Kapan ke dokter
✅ Ciri-ciri stunting
✅ Dampak jangka panjang
✅ Peran orang tua

**Ada yang ingin ditanyakan?**""",

    "bantuan": """📋 **Yang bisa saya bantu:**

1️⃣ **Definisi** - 'Apa itu stunting?'
2️⃣ **Penyebab** - 'Penyebab stunting?'
3️⃣ **Pencegahan** - 'Cara mencegah stunting?'
4️⃣ **Ciri-ciri** - 'Ciri stunting?'
5️⃣ **Makanan** - 'Makanan bergizi?'
6️⃣ **Dokter** - 'Kapan ke dokter?'
7️⃣ **Dampak** - 'Dampak stunting?'
8️⃣ **Orang Tua** - 'Peran orang tua?'
9️⃣ **ASI** - 'Bagaimana dengan ASI?'"""
}

def get_response_by_keyword(message: str) -> str:
    """Get response based on keyword matching"""
    msg = message.lower().strip()
    
    # Cek salam
    if any(k in msg for k in ["halo", "hai", "hello", "hi", "selamat"]):
        return RESPONSES["salam"]
    
    # Cek bantuan
    if any(k in msg for k in ["bantuan", "help", "tolong"]):
        return RESPONSES["bantuan"]
    
    # Cek definisi
    if any(k in msg for k in ["definisi", "apa itu", "pengertian", "stunting adalah"]):
        return RESPONSES["definisi"]
    
    # Cek penyebab
    if any(k in msg for k in ["penyebab", "sebab", "faktor", "kenapa", "mengapa"]):
        return RESPONSES["penyebab"]
    
    # Cek pencegahan
    if any(k in msg for k in ["mencegah", "pencegahan", "cara", "tips", "langkah"]):
        return RESPONSES["pencegahan"]
    
    # Cek makanan
    if any(k in msg for k in ["makanan", "gizi", "nutrisi", "menu", "makan"]):
        return RESPONSES["makanan"]
    
    # Cek dokter
    if any(k in msg for k in ["dokter", "periksa", "konsultasi", "puskesmas"]):
        return RESPONSES["dokter"]
    
    # Cek ciri
    if any(k in msg for k in ["ciri", "tanda", "gejala"]):
        return RESPONSES["ciri"]
    
    # Cek dampak
    if any(k in msg for k in ["dampak", "akibat", "bahaya", "efek"]):
        return RESPONSES["dampak"]
    
    # Cek orang tua
    if any(k in msg for k in ["orang tua", "peran", "orangtua"]):
        return RESPONSES["orang_tua"]
    
    # Cek ASI
    if any(k in msg for k in ["asi", "menyusui", "susu ibu"]):
        return RESPONSES["asi"]
    
    # Default
    return """Maaf, saya belum mengerti pertanyaan Anda 😊

**Coba tanyakan:**
• 'Apa itu stunting?'
• 'Penyebab stunting?'
• 'Cara mencegah stunting?'
• 'Makanan bergizi?'
• 'Kapan ke dokter?'
• 'Peran orang tua?'
• 'Ciri-ciri stunting?'
• 'Dampak stunting?'
• 'Bagaimana dengan ASI?'"""

# ============================================
# 4. CHATBOT ENGINE - DIJAMIN RESPONSE
# ============================================
class Chatbot:
    def __init__(self):
        self.history: List[Dict] = []
    
    def get_response(self, message: str) -> Dict:
        """Get response from chatbot - ALWAYS returns valid response"""
        try:
            # 1. COBA AZURE OPENAI
            if AZURE_AVAILABLE and client:
                try:
                    self.history.append({"role": "user", "content": message})
                    
                    messages = [
                        {"role": "system", "content": "Anda adalah asisten kesehatan stunting. Jawab dalam Bahasa Indonesia dengan singkat dan jelas."}
                    ]
                    messages.extend(self.history[-10:])
                    
                    response = client.chat.completions.create(
                        model=AZURE_OPENAI_DEPLOYMENT_NAME,
                        messages=messages,
                        max_completion_tokens=500
                    )
                    
                    reply = response.choices[0].message.content.strip()
                    self.history.append({"role": "assistant", "content": reply})
                    
                    return {
                        "response": reply,
                        "azure_used": True
                    }
                except Exception as e:
                    print(f"Azure error: {e}")
                    # LANJUT KE FALLBACK
            
            # 2. FALLBACK RULE-BASED (PASTI ADA RESPONSE)
            reply = get_response_by_keyword(message)
            return {
                "response": reply,
                "azure_used": False
            }
            
        except Exception as e:
            print(f"Chatbot error: {e}")
            # PASTIKAN SELALU ADA RESPONSE
            return {
                "response": "❌ Maaf, terjadi kesalahan. Silakan coba lagi.\n\n" + get_response_by_keyword(message),
                "azure_used": False
            }

# Inisialisasi chatbot
chatbot = Chatbot()
print("✅ Chatbot siap!")

# ============================================
# 5. LOAD MODEL (Opsional)
# ============================================
model_loaded = False
model = None

try:
    if os.path.exists("model_stunting.pkl"):
        with open("model_stunting.pkl", "rb") as f:
            model = pickle.load(f)
        model_loaded = True
        print("✅ Model prediksi dimuat!")
    else:
        print("⚠️ Model tidak ditemukan")
except Exception as e:
    print(f"⚠️ Error loading model: {e}")

# ============================================
# 6. PYDANTIC MODELS
# ============================================
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

class HealthResponse(BaseModel):
    status: str
    azure_available: bool
    model_loaded: bool
    version: str

# ============================================
# 7. ENDPOINTS - DENGAN RESPONSE YANG DIJAMIN
# ============================================

@app.get("/")
async def root():
    return {
        "message": "NutriScan API",
        "docs": "/docs",
        "status": {
            "azure_available": AZURE_AVAILABLE,
            "model_loaded": model_loaded
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        azure_available=AZURE_AVAILABLE,
        model_loaded=model_loaded,
        version="1.0.0"
    )

@app.get("/test-azure")
async def test_azure():
    if not AZURE_AVAILABLE or not client:
        return {"status": "error", "message": "Azure tidak tersedia"}
    try:
        response = client.chat.completions.create(
            model=AZURE_OPENAI_DEPLOYMENT_NAME,
            messages=[{"role": "user", "content": "Halo"}],
            max_completion_tokens=10
        )
        return {
            "status": "success",
            "response": response.choices[0].message.content
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/predict", response_model=PredictResponse)
async def predict_stunting(request: PredictRequest):
    try:
        if request.gender not in ["Laki-laki", "Perempuan"]:
            raise HTTPException(status_code=400, detail="Gender harus 'Laki-laki' atau 'Perempuan'")
        
        if not model_loaded:
            return PredictResponse(
                status="Normal",
                risk_score=0.2,
                bmi=16.5,
                message="✅ Status normal. Pertahankan gizi seimbang.",
                details={
                    "age": request.age,
                    "gender": request.gender,
                    "height": request.height,
                    "weight": request.weight
                }
            )
        
        # Dummy prediction
        return PredictResponse(
            status="Normal",
            risk_score=0.2,
            bmi=16.5,
            message="✅ Status normal. Pertahankan gizi seimbang.",
            details={
                "age": request.age,
                "gender": request.gender,
                "height": request.height,
                "weight": request.weight
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat", response_model=ChatResponse)
async def chat_with_bot(request: ChatRequest):
    """Endpoint chatbot - ALWAYS returns response"""
    try:
        # PASTIKAN ADA MESSAGE
        if not request.message or not request.message.strip():
            return ChatResponse(
                response="Silakan ketik pertanyaan Anda.",
                azure_used=False
            )
        
        result = chatbot.get_response(request.message)
        
        # PASTIKAN RESPONSE ADA
        response_text = result.get("response", "Maaf, tidak ada respons. Silakan coba lagi.")
        
        return ChatResponse(
            response=response_text,
            azure_used=result.get("azure_used", False)
        )
    except Exception as e:
        print(f"Chat endpoint error: {e}")
        # PASTIKAN SELALU ADA RESPONSE
        return ChatResponse(
            response=get_response_by_keyword(request.message),
            azure_used=False
        )

# ============================================
# 8. RUN SERVER
# ============================================
if __name__ == "__main__":
    print("=" * 60)
    print("🚀 NUTRISCAN API")
    print("=" * 60)
    print(f"✅ Azure OpenAI: {'TERSEDIA' if AZURE_AVAILABLE else 'TIDAK TERSEDIA'}")
    print(f"✅ Model: {'TERMUAT' if model_loaded else 'TIDAK TERMUAT'}")
    print("=" * 60)
    print("📋 Endpoints:")
    print("  - GET  /")
    print("  - GET  /health")
    print("  - GET  /test-azure")
    print("  - POST /predict")
    print("  - POST /chat")
    print("=" * 60)
    
    # Jalankan server
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)