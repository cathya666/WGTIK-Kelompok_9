"""
NutriScan - Chatbot Engine dengan Azure OpenAI (Full)
Semua respons menggunakan Azure OpenAI dengan context handling
"""
import re
import os
from typing import Dict, List, Optional
from datetime import datetime
from openai import AzureOpenAI
from dotenv import load_dotenv

# ==========================================
# 1. LOAD .env DAN INISIALISASI AZURE OPENAI
# ==========================================
load_dotenv()

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-10-21",
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
)

# ==========================================
# 2. CONTEXT HANDLING
# ==========================================
class ChatContext:
    def __init__(self, max_history: int = 10):
        self.history: List[Dict] = []
        self.max_history = max_history

    def add_message(self, role: str, message: str):
        self.history.append({
            "role": role,
            "message": message,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })
        # Simpan maksimal 10 percakapan (5 user + 5 bot)
        if len(self.history) > self.max_history * 2:
            self.history = self.history[-(self.max_history * 2):]

    def reset(self):
        self.history = []

    def get_history(self) -> List[Dict]:
        return self.history

    def get_conversation_context(self) -> List[Dict]:
        """Format history untuk OpenAI API"""
        messages = []
        for entry in self.history:
            if entry["role"] == "user":
                messages.append({"role": "user", "content": entry["message"]})
            elif entry["role"] == "bot":
                messages.append({"role": "assistant", "content": entry["message"]})
        return messages


# ==========================================
# 3. SYSTEM PROMPT - DIPERKUAT UNTUK SEMUA TOPIK
# ==========================================
SYSTEM_PROMPT = """Anda adalah NutriScan Bot, asisten edukasi kesehatan yang ahli dalam stunting dan gizi anak.

Anda HARUS menjawab pertanyaan dengan INFORMATIF, LENGKAP, dan EMPATIK dalam Bahasa Indonesia.

TOPIK YANG HARUS DIKUASAI (jawab detail untuk semua):
1. Definisi Stunting - Apa itu stunting, penjelasan lengkap
2. Penyebab Stunting - Semua faktor penyebab (gizi, sanitasi, dll)
3. Pencegahan Stunting - Langkah-langkah konkret
4. Ciri-ciri Stunting - Tanda fisik dan perilaku
5. Makanan Bergizi - Rekomendasi makanan untuk cegah stunting
6. Dampak Stunting - Jangka pendek dan panjang
7. Peran Orang Tua - Apa yang harus dilakukan
8. Kapan ke Dokter - Tanda-tanda darurat dan konsultasi
9. Deteksi Dini - Cara mendeteksi stunting di rumah

RULES:
- Selalu jawab dalam Bahasa Indonesia yang baik dan mudah dimengerti
- Gunakan format poin-poin (bullet points) untuk memudahkan pembacaan
- Berikan saran yang praktis dan aplikatif
- Jika pertanyaan di luar topik, arahkan ke topik stunting
- Tanyakan kembali jika ada yang kurang jelas
- Jangan menjawab dengan "maaf saya tidak tahu" - coba bantu sebisa mungkin

Contoh jawaban untuk 'Kapan harus ke dokter?':
"Segera bawa anak ke dokter atau puskesmas jika:
1. Tinggi badan tidak naik selama 2 bulan berturut-turut
2. Berat badan turun atau stagnan > 1 bulan
3. Anak sering sakit (diare, batuk, demam berulang)
4. Anak tampak lemas, tidak aktif, atau nafsu makan buruk
5. Hasil deteksi menunjukkan Stunting

Pemantauan rutin ke posyandu setiap bulan juga sangat dianjurkan!"

Mulai percakapan dengan sapaan ramah jika pengguna menyapa."""


# ==========================================
# 4. MAIN CHATBOT CLASS - FULL AZURE OPENAI
# ==========================================
class StuntingChatbot:
    def __init__(self):
        self.context = ChatContext(max_history=10)
        self.conversation_started = False

    def get_response(self, message: str) -> Dict:
        """Proses pesan user dan dapatkan respons dari Azure OpenAI"""
        if not message or not message.strip():
            return {
                "response": "Mohon ketikkan pertanyaan Anda.",
                "intent": "empty",
                "confidence": 0.0
            }

        # Bersihkan pesan
        clean_message = message.strip()

        # Tambahkan pesan user ke context
        self.context.add_message("user", clean_message)

        # Siapkan messages untuk OpenAI
        messages = []

        # Tambahkan system prompt
        messages.append({"role": "system", "content": SYSTEM_PROMPT})

        # Tambahkan history (max 5 percakapan terakhir)
        history_messages = self.context.get_conversation_context()
        # Ambil 5 percakapan terakhir (10 messages)
        if len(history_messages) > 10:
            history_messages = history_messages[-10:]
        messages.extend(history_messages)

        # Jika hanya ada system prompt + 1 user message, tambahkan pengantar
        if len(messages) <= 2:
            messages.append({
                "role": "assistant", 
                "content": "Halo! Saya NutriScan Bot. Silakan tanyakan tentang stunting, gizi anak, atau tumbuh kembang."
            })

        try:
            # Panggil Azure OpenAI
            response = client.chat.completions.create(
                model="gpt4o-chat",  # atau model yang Anda gunakan
                messages=messages,
                temperature=0.7,
                max_tokens=500,
                top_p=0.9
            )

            response_text = response.choices[0].message.content.strip()

        except Exception as e:
            print(f"Error Azure OpenAI: {e}")
            response_text = (
                "Maaf, sistem saya sedang mengalami gangguan. "
                "Silakan coba lagi dalam beberapa saat. "
                "Jika masih bermasalah, Anda bisa berkonsultasi langsung ke puskesmas terdekat."
            )

        # Simpan respons bot ke context
        self.context.add_message("bot", response_text)

        return {
            "response": response_text,
            "intent": "azure_openai",
            "confidence": 1.0
        }

    def reset_context(self):
        """Reset percakapan"""
        self.context.reset()
        self.conversation_started = False

    def get_history(self) -> List[Dict]:
        """Dapatkan history percakapan"""
        return self.context.get_history()

    def get_formatted_history(self) -> List[Dict]:
        """Dapatkan history dalam format untuk ditampilkan"""
        history = []
        for entry in self.context.get_history():
            history.append({
                "role": entry["role"],
                "message": entry["message"],
                "timestamp": entry["timestamp"]
            })
        return history


# ==========================================
# 5. FUNGSI UTILITY (untuk integrasi dengan frontend)
# ==========================================

# Inisialisasi singleton chatbot
_chatbot = None

def get_chatbot() -> StuntingChatbot:
    """Get atau create instance chatbot (singleton)"""
    global _chatbot
    if _chatbot is None:
        _chatbot = StuntingChatbot()
    return _chatbot

def process_message(message: str) -> Dict:
    """Fungsi utama untuk memproses pesan dari frontend"""
    chatbot = get_chatbot()
    return chatbot.get_response(message)

def reset_conversation() -> Dict:
    """Reset percakapan"""
    chatbot = get_chatbot()
    chatbot.reset_context()
    return {"status": "success", "message": "Percakapan direset"}

def get_conversation_history() -> List[Dict]:
    """Dapatkan history percakapan"""
    chatbot = get_chatbot()
    return chatbot.get_formatted_history()


# ==========================================
# 6. TESTING (optional - jalankan jika file di-run langsung)
# ==========================================
if __name__ == "__main__":
    # Test chatbot
    bot = StuntingChatbot()
    
    test_questions = [
        "Halo",
        "Apa itu stunting?",
        "Kapan harus ke dokter?",
        "Makanan apa yang baik untuk anak?",
        "Terima kasih"
    ]
    
    print("=" * 50)
    print("TESTING CHATBOT (Full Azure OpenAI)")
    print("=" * 50)
    
    for q in test_questions:
        print(f"\nUser: {q}")
        result = bot.get_response(q)
        print(f"Bot: {result['response']}")
        print(f"Intent: {result['intent']}")
        print("-" * 40)