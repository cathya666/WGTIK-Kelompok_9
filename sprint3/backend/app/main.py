# ============================================
# NUTRISCAN API - MAIN.PY
# FastAPI Backend untuk Deteksi Stunting + Chatbot
# ============================================

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
import pickle
import pandas as pd
import numpy as np
from pathlib import Path

# ============================================
# 1. INISIALISASI APP
# ============================================
app = FastAPI(
    title="NutriScan API",
    description="API untuk deteksi stunting pada balita dan chatbot edukasi",
    version="1.0.0"
)

# CORS Middleware (agar frontend bisa akses)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Bisa diganti dengan domain frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================
# 2. LOAD MODEL & PREPROCESSING TOOLS
# ============================================
MODEL_PATH = Path("model")

# Load model
with open(MODEL_PATH / "model_stunting.pkl", "rb") as f:
    model = pickle.load(f)

# Load scaler
with open(MODEL_PATH / "scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Load gender encoder
with open(MODEL_PATH / "gender_encoder.pkl", "rb") as f:
    gender_encoder = pickle.load(f)

print("✅ Model, scaler, dan gender encoder berhasil dimuat!")

# ============================================
# 3. PYDANTIC MODELS (Request/Response)
# ============================================

class PredictRequest(BaseModel):
    """Request untuk endpoint /predict"""
    age: float = Field(..., description="Usia dalam bulan", ge=0, le=59)
    gender: str = Field(..., description="Jenis kelamin: Laki-laki atau Perempuan")
    height: float = Field(..., description="Tinggi badan dalam cm", ge=45, le=120)
    weight: float = Field(..., description="Berat badan dalam kg", ge=2, le=25)
    
    class Config:
        json_schema_extra = {
            "example": {
                "age": 24,
                "gender": "Laki-laki",
                "height": 80.5,
                "weight": 10.2
            }
        }

class PredictResponse(BaseModel):
    """Response dari endpoint /predict"""
    status: str  # "Normal" atau "Stunted"
    risk_score: float  # Probabilitas stunting (0-1)
    bmi: float  # Body Mass Index
    message: str  # Rekomendasi/saran
    details: Optional[Dict[str, Any]] = None

class ChatRequest(BaseModel):
    """Request untuk endpoint /chat"""
    message: str = Field(..., description="Pesan user untuk chatbot")

class ChatResponse(BaseModel):
    """Response dari endpoint /chat"""
    response: str
    intent: str

class HealthResponse(BaseModel):
    """Response dari endpoint /health"""
    status: str
    model_loaded: bool
    version: str

# ============================================
# 4. FUNGSI PREPROCESSING
# ============================================

def preprocess_input(age: float, gender: str, height: float, weight: float) -> np.ndarray:
    """
    Preprocessing input sebelum diprediksi model
    """
    # Encode gender
    gender_encoded = gender_encoder.transform([gender])[0]
    
    # Hitung BMI
    height_m = height / 100
    bmi = weight / (height_m ** 2)
    
    # Buat dataframe
    df_input = pd.DataFrame({
        'Age': [age],
        'Gender': [gender_encoded],
        'Height': [height],
        'Weight': [weight],
        'BMI': [bmi]
    })
    
    # Scaling
    X_scaled = scaler.transform(df_input)
    
    return X_scaled, bmi

def get_recommendation(status: str, bmi: float = None) -> str:
    """
    Memberikan rekomendasi berdasarkan hasil prediksi
    """
    if status == "Stunted":
        return "⚠️ Anak terindikasi stunting. Segera konsultasikan ke dokter atau puskesmas terdekat untuk penanganan lebih lanjut. Pastikan asupan gizi seimbang dan lakukan pemantauan pertumbuhan rutin setiap bulan."
    else:
        if bmi and bmi < 14:
            return "✅ Status gizi normal, namun berat badan kurang. Tingkatkan asupan protein, karbohidrat, dan lemak sehat. Lanjutkan pemantauan rutin."
        elif bmi and bmi > 18:
            return "✅ Status gizi normal, namun berat badan berlebih. Jaga keseimbangan asupan gizi dan tingkatkan aktivitas fisik sesuai usia."
        else:
            return "✅ Status gizi normal. Pertahankan asupan gizi seimbang dan lakukan pemantauan pertumbuhan rutin setiap bulan."

# ============================================
# 5. CHATBOT ENGINE (Simple Rule-based)
# ============================================

class StuntingChatbot:
    """
    Chatbot sederhana untuk edukasi stunting
    """
    
    def __init__(self):
        # Intent keywords mapping
        self.intents = {
            "definisi_stunting": [
                "apa itu stunting", "stunting adalah", "definisi stunting", 
                "pengertian stunting", "stunting itu apa"
            ],
            "penyebab_stunting": [
                "penyebab stunting", "kenapa anak stunting", "faktor stunting",
                "sebab stunting", "apa yang menyebabkan stunting"
            ],
            "pencegahan_stunting": [
                "mencegah stunting", "cegah stunting", "cara pencegahan",
                "bagaimana mencegah stunting", "pencegahan stunting"
            ],
            "ciri_stunting": [
                "ciri stunting", "tanda stunting", "gejala stunting",
                "bagaimana ciri anak stunting"
            ],
            "makanan_stunting": [
                "makanan untuk stunting", "gizi stunting", "nutrisi stunting",
                "makanan cegah stunting", "menu stunting"
            ],
            "dampak_stunting": [
                "dampak stunting", "akibat stunting", "efek stunting",
                "bahaya stunting"
            ],
            "peran_orang_tua": [
                "peran orang tua", "apa yang harus dilakukan orang tua",
                "peran keluarga", "orang tua bisa apa"
            ],
            "salam": [
                "halo", "hai", "selamat pagi", "selamat siang", 
                "hy", "hello", "hi", "assalamualaikum"
            ],
            "terima_kasih": [
                "terima kasih", "thanks", "makasih", "thank you"
            ],
            "bantuan": [
                "help", "tolong", "bantuan", "apa yang bisa kamu lakukan",
                "fitur", "command"
            ]
        }
        
        # Response templates
        self.responses = {
            "definisi_stunting": (
                "📚 **Apa itu Stunting?**\n\n"
                "Stunting adalah kondisi gagal tumbuh pada anak balita (0-59 bulan) "
                "akibat kekurangan gizi kronis, terutama dalam 1000 Hari Pertama Kehidupan (HPK). "
                "Anak stunting memiliki tinggi badan di bawah standar usianya menurut kurva pertumbuhan WHO."
            ),
            "penyebab_stunting": (
                "🔍 **Penyebab Stunting:**\n\n"
                "1. Kurangnya asupan gizi kronis (protein, vitamin, mineral)\n"
                "2. ASI tidak eksklusif 6 bulan\n"
                "3. MPASI yang kurang bergizi dan tidak tepat waktu\n"
                "4. Infeksi berulang (diare, ISPA)\n"
                "5. Sanitasi lingkungan yang buruk\n"
                "6. Akses terbatas ke layanan kesehatan\n"
                "7. Pendidikan ibu yang rendah\n"
                "8. Kemiskinan dan ketahanan pangan buruk"
            ),
            "pencegahan_stunting": (
                "🛡️ **Cara Mencegah Stunting:**\n\n"
                "1. ✅ ASI eksklusif selama 6 bulan pertama\n"
                "2. ✅ Berikan MPASI bergizi seimbang (protein hewani, sayur, buah)\n"
                "3. ✅ Penuhi gizi ibu hamil dengan tablet tambah darah\n"
                "4. ✅ Pantau pertumbuhan anak setiap bulan di posyandu\n"
                "5. ✅ Jaga kebersihan lingkungan dan sanitasi\n"
                "6. ✅ Lengkapi imunisasi dasar anak\n"
                "7. ✅ Berikan stimulasi tumbuh kembang yang tepat"
            ),
            "ciri_stunting": (
                "👶 **Ciri-ciri Anak Stunting:**\n\n"
                "1. Tinggi badan di bawah standar usianya\n"
                "2. Pertumbuhan melambat (tidak naik berat/tinggi)\n"
                "3. Wajah tampak lebih muda dari usianya\n"
                "4. Keterlambatan perkembangan motorik dan kognitif\n"
                "5. Lebih pendek dari anak seusianya\n"
                "6. Mudah sakit dan daya tahan tubuh rendah"
            ),
            "makanan_stunting": (
                "🍳 **Makanan untuk Cegah & Atasi Stunting:**\n\n"
                "🔹 **Protein Hewani (WAJIB setiap hari):**\n"
                "   - Telur (1 butir/hari)\n"
                "   - Ikan (teri, salmon, kembung)\n"
                "   - Ayam, hati ayam\n"
                "   - Susu dan olahannya\n\n"
                "🔹 **Karbohidrat:** Nasi, ubi, jagung, kentang\n\n"
                "🔹 **Sayur & Buah:** Bayam, wortel, brokoli, pepaya, pisang\n\n"
                "🔹 **Lemak Sehat:** Minyak ikan, alpukat, santan\n\n"
                "💡 **Tips:** Berikan variasi menu dan tekstur sesuai usia anak."
            ),
            "dampak_stunting": (
                "⚠️ **Dampak Stunting Jangka Panjang:**\n\n"
                "1. 🧠 Menurunnya kemampuan kognitif dan prestasi belajar\n"
                "2. 📉 Produktivitas rendah saat dewasa\n"
                "3. 💰 Risiko kemiskinan yang lebih tinggi\n"
                "4. 🩺 Meningkatnya risiko penyakit tidak menular (diabetes, hipertensi)\n"
                "5. 📏 Postur tubuh pendek saat dewasa"
            ),
            "peran_orang_tua": (
                "👨‍👩‍👧 **Peran Orang Tua Mencegah Stunting:**\n\n"
                "1. Memastikan asupan gizi seimbang setiap hari\n"
                "2. Memberikan ASI eksklusif dan MPASI bergizi\n"
                "3. Rutin memantau pertumbuhan ke posyandu\n"
                "4. Menjaga kebersihan lingkungan dan sanitasi\n"
                "5. Memberikan stimulasi tumbuh kembang\n"
                "6. Segera konsultasi jika ada tanda-tanda stunting"
            ),
            "salam": (
                "Halo! 👋 Saya NutriScan Bot. Saya di sini untuk membantu Anda.\n\n"
                "📋 **Yang bisa saya bantu:**\n"
                "• Deteksi stunting (gunakan form di samping)\n"
                "• Informasi tentang stunting\n"
                "• Penyebab dan pencegahan stunting\n"
                "• Rekomendasi makanan bergizi\n\n"
                "Ada yang ingin ditanyakan tentang stunting?"
            ),
            "terima_kasih": (
                "Sama-sama! 😊 Senang bisa membantu.\n\n"
                "Ada yang ingin ditanyakan lagi? Atau Anda bisa mencoba fitur deteksi stunting di form input data anak."
            ),
            "bantuan": (
                "📋 **Yang bisa saya lakukan:**\n\n"
                "1️⃣ **Tanya tentang stunting** - 'Apa itu stunting?'\n"
                "2️⃣ **Penyebab stunting** - 'Kenapa anak bisa stunting?'\n"
                "3️⃣ **Pencegahan stunting** - 'Bagaimana mencegah stunting?'\n"
                "4️⃣ **Ciri-ciri stunting** - 'Ciri anak stunting?'\n"
                "5️⃣ **Makanan bergizi** - 'Makanan untuk cegah stunting?'\n"
                "6️⃣ **Dampak stunting** - 'Apa dampak stunting?'\n\n"
                "✨ **Plus:** Gunakan form di samping untuk deteksi stunting berdasarkan data anak Anda!"
            ),
            "unknown": (
                "Maaf, saya belum mengerti pertanyaan Anda 🙏\n\n"
                "Coba tanyakan:\n"
                "• 'Apa itu stunting?'\n"
                "• 'Penyebab stunting?'\n"
                "• 'Cara mencegah stunting?'\n"
                "• 'Makanan untuk stunting?'\n\n"
                "Atau coba ketik 'bantuan' untuk melihat semua fitur."
            )
        }
    
    def get_response(self, message: str) -> Dict[str, str]:
        """Proses pesan user dan kembalikan respons"""
        message_lower = message.lower().strip()
        
        # Cari intent berdasarkan keyword
        for intent, keywords in self.intents.items():
            if any(keyword in message_lower for keyword in keywords):
                return {
                    "response": self.responses[intent],
                    "intent": intent
                }
        
        # Jika tidak ada intent yang cocok
        return {
            "response": self.responses["unknown"],
            "intent": "unknown"
        }

# Inisialisasi chatbot
chatbot = StuntingChatbot()
print("✅ Chatbot siap digunakan!")

# ============================================
# 6. ENDPOINTS
# ============================================

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to NutriScan API",
        "documentation": "/docs",
        "endpoints": ["/predict", "/chat", "/health"]
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Cek status server dan model"""
    return HealthResponse(
        status="healthy",
        model_loaded=True,
        version="1.0.0"
    )

@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
async def predict_stunting(request: PredictRequest):
    """
    Endpoint untuk prediksi stunting berdasarkan:
    - Usia (bulan)
    - Jenis kelamin (Laki-laki/Perempuan)
    - Tinggi badan (cm)
    - Berat badan (kg)
    """
    try:
        # Validasi gender
        if request.gender not in ["Laki-laki", "Perempuan"]:
            raise HTTPException(
                status_code=400,
                detail="Gender harus 'Laki-laki' atau 'Perempuan'"
            )
        
        # Preprocessing
        X_scaled, bmi = preprocess_input(
            age=request.age,
            gender=request.gender,
            height=request.height,
            weight=request.weight
        )
        
        # Prediksi
        prediction = model.predict(X_scaled)[0]
        probability = model.predict_proba(X_scaled)[0]
        
        # Interpretasi
        if prediction == 1:
            status = "Stunted"
            risk_score = float(probability[1])
        else:
            status = "Normal"
            risk_score = float(probability[0])
        
        # Dapatkan rekomendasi
        message = get_recommendation(status, bmi)
        
        # Details tambahan
        details = {
            "age_months": request.age,
            "gender": request.gender,
            "height_cm": request.height,
            "weight_kg": request.weight,
            "bmi": round(bmi, 2)
        }
        
        return PredictResponse(
            status=status,
            risk_score=risk_score,
            bmi=round(bmi, 2),
            message=message,
            details=details
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/chat", response_model=ChatResponse, tags=["Chatbot"])
async def chat_with_bot(request: ChatRequest):
    """
    Endpoint untuk chatbot edukasi stunting
    """
    try:
        response = chatbot.get_response(request.message)
        return ChatResponse(
            response=response["response"],
            intent=response["intent"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chatbot error: {str(e)}")

# ============================================
# 7. RUN SERVER (untuk debugging)
# ============================================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)