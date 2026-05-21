# chatbot.py
# NutriScan - Chatbot Engine dengan NLP sederhana + Context Handling

import re
from typing import Dict, List, Optional, Tuple
from datetime import datetime


class ChatContext:
    def __init__(self, max_history: int = 5):
        self.history: List[Dict] = []
        self.last_intent: Optional[str] = None
        self.max_history = max_history

    def add_message(self, role: str, message: str, intent: Optional[str] = None):
        self.history.append({
            "role": role,
            "message": message,
            "intent": intent,
            "timestamp": datetime.now().strftime("%H:%M:%S")
        })
        if len(self.history) > self.max_history * 2:
            self.history = self.history[-(self.max_history * 2):]
        if intent:
            self.last_intent = intent

    def reset(self):
        self.history = []
        self.last_intent = None

    def get_history(self) -> List[Dict]:
        return self.history


class SimpleNLP:
    SLANG_MAP = {
        "gak": "tidak", "ga": "tidak", "ngga": "tidak", "nggak": "tidak",
        "udah": "sudah", "udh": "sudah",
        "gimana": "bagaimana", "gmn": "bagaimana",
        "kenapa": "mengapa", "knp": "mengapa",
        "org": "orang", "yg": "yang", "dgn": "dengan",
        "utk": "untuk", "krn": "karena", "karna": "karena",
        "spt": "seperti", "tp": "tapi", "tpi": "tapi",
        "sy": "saya", "aku": "saya",
        "nutrisi": "gizi", "mkn": "makanan",
        "hrs": "harus",
    }

    @staticmethod
    def normalize(text: str) -> str:
        text = text.lower().strip()
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text)
        words = text.split()
        words = [SimpleNLP.SLANG_MAP.get(w, w) for w in words]
        return " ".join(words)

    @staticmethod
    def tokenize(text: str) -> List[str]:
        return re.findall(r"\b\w+\b", SimpleNLP.normalize(text))

    @staticmethod
    def get_all_phrases(text: str) -> List[str]:
        tokens = SimpleNLP.tokenize(text)
        phrases = list(tokens)
        for n in range(2, 5):
            for i in range(len(tokens) - n + 1):
                phrases.append(" ".join(tokens[i:i+n]))
        return phrases


class IntentClassifier:
    def __init__(self):
        self.intent_keywords: Dict[str, List[Tuple[str, int]]] = {
            "salam": [
                ("halo", 2), ("hai", 2), ("hi", 2), ("hello", 2),
                ("selamat pagi", 3), ("selamat siang", 3),
                ("selamat sore", 3), ("selamat malam", 3),
                ("assalamualaikum", 3), ("permisi", 1), ("hei", 1),
            ],
            "terima_kasih": [
                ("terima kasih", 3), ("thanks", 2), ("makasih", 2),
                ("thank you", 3), ("thx", 2), ("tq", 2), ("trims", 2),
            ],
            "bantuan": [
                ("bantuan", 2), ("help", 2), ("tolong bantu", 3),
                ("apa yang bisa kamu lakukan", 4), ("fitur", 2),
                ("panduan", 2), ("apa saja yang bisa", 3), ("command", 2),
            ],
            "definisi_stunting": [
                ("apa itu stunting", 5), ("stunting adalah", 4),
                ("definisi stunting", 5), ("pengertian stunting", 5),
                ("stunting itu apa", 5), ("arti stunting", 4),
                ("jelaskan stunting", 4), ("apa stunting", 3), ("stunting", 1),
            ],
            "penyebab_stunting": [
                ("penyebab stunting", 5), ("kenapa anak stunting", 5),
                ("faktor stunting", 4), ("faktor risiko stunting", 5),
                ("sebab stunting", 4), ("apa yang menyebabkan stunting", 5),
                ("mengapa stunting", 4), ("apa penyebab stunting", 5),
                ("penyebab", 1),
            ],
            "pencegahan_stunting": [
                ("mencegah stunting", 5), ("cegah stunting", 5),
                ("cara pencegahan stunting", 5),
                ("bagaimana mencegah stunting", 5),
                ("pencegahan stunting", 5), ("tips stunting", 3),
                ("cara mengatasi stunting", 4),
                ("mencegah", 1), ("pencegahan", 1),
            ],
            "ciri_stunting": [
                ("ciri stunting", 5), ("tanda stunting", 5),
                ("gejala stunting", 5), ("ciri ciri stunting", 5),
                ("tanda tanda stunting", 5),
                ("anak stunting seperti apa", 4),
                ("bagaimana ciri anak stunting", 5),
                ("ciri", 1), ("tanda", 1), ("gejala", 1),
            ],
            "makanan_stunting": [
                ("makanan untuk stunting", 5), ("makanan cegah stunting", 5),
                ("makanan anak stunting", 5), ("gizi stunting", 4),
                ("makanan bergizi anak", 4), ("nutrisi stunting", 4),
                ("menu stunting", 4), ("saran gizi", 3),
                ("asupan stunting", 4), ("makanan baik untuk anak", 4),
                ("makanan", 1), ("gizi", 1),
            ],
            "dampak_stunting": [
                ("dampak stunting", 5), ("akibat stunting", 5),
                ("efek stunting", 4), ("bahaya stunting", 4),
                ("dampak jangka panjang stunting", 5),
                ("risiko stunting", 4),
                ("dampak", 1), ("akibat", 1), ("bahaya", 1),
            ],
            "peran_orang_tua": [
                ("peran orang tua", 5), ("peran keluarga stunting", 5),
                ("apa yang harus dilakukan orang tua", 5),
                ("orang tua bisa apa", 4), ("peran ibu stunting", 4),
                ("tips orang tua stunting", 4), ("orang tua", 2),
            ],
            "kapan_ke_dokter": [
                ("kapan ke dokter", 5), ("kapan harus ke dokter", 5),
                ("ke dokter", 3), ("konsultasi dokter", 4),
                ("kapan periksa", 4), ("kapan ke puskesmas", 4),
                ("tanda harus ke dokter", 5), ("segera ke dokter", 4),
            ],
        }
        self.CONFIDENCE_THRESHOLD = 2

    def classify(self, text: str, context=None) -> Tuple[str, float]:
        phrases_set = set(SimpleNLP.get_all_phrases(text))
        normalized  = SimpleNLP.normalize(text)
        intent_scores: Dict[str, float] = {}
        for intent, keywords in self.intent_keywords.items():
            score = 0.0
            for keyword, weight in keywords:
                if keyword in phrases_set or keyword in normalized:
                    score += weight
            if score > 0:
                intent_scores[intent] = score
        if not intent_scores:
            if context and context.last_intent:
                resolved = self._resolve_with_context(text, context.last_intent)
                if resolved:
                    return resolved, 0.3
            return "unknown", 0.0
        best_intent = max(intent_scores, key=lambda k: intent_scores[k])
        best_score  = intent_scores[best_intent]
        if best_score < self.CONFIDENCE_THRESHOLD:
            if context and context.last_intent:
                resolved = self._resolve_with_context(text, context.last_intent)
                if resolved:
                    return resolved, 0.3
            return "unknown", 0.0
        return best_intent, min(best_score / 10.0, 1.0)

    def _resolve_with_context(self, text: str, last_intent: str) -> Optional[str]:
        follow_up_words = {"kenapa", "mengapa", "lanjut", "terus",
                           "selanjutnya", "lebih", "dan"}
        tokens = set(SimpleNLP.tokenize(text))
        if tokens & follow_up_words:
            return last_intent
        return None


class ResponseGenerator:
    def __init__(self):
        self.responses = {
            "salam": (
                "Halo! Saya NutriScan Bot, asisten edukasi stunting.\n\n"
                "Yang bisa saya bantu:\n"
                "- Informasi tentang stunting\n"
                "- Penyebab dan pencegahan stunting\n"
                "- Ciri-ciri anak stunting\n"
                "- Rekomendasi makanan bergizi\n"
                "- Dampak stunting\n\n"
                "Ada yang ingin ditanyakan?"
            ),
            "terima_kasih": (
                "Sama-sama! Senang bisa membantu.\n\n"
                "Ada pertanyaan lain seputar stunting?"
            ),
            "bantuan": (
                "Yang bisa saya lakukan:\n\n"
                "1. Tanya tentang stunting - 'Apa itu stunting?'\n"
                "2. Penyebab stunting - 'Kenapa anak bisa stunting?'\n"
                "3. Pencegahan stunting - 'Bagaimana mencegah stunting?'\n"
                "4. Ciri-ciri stunting - 'Ciri anak stunting?'\n"
                "5. Makanan bergizi - 'Makanan untuk cegah stunting?'\n"
                "6. Dampak stunting - 'Apa dampak stunting?'\n"
                "7. Peran orang tua - 'Apa yang bisa dilakukan orang tua?'\n\n"
                "Gunakan form di samping untuk deteksi stunting!"
            ),
            "definisi_stunting": (
                "Apa itu Stunting?\n\n"
                "Stunting adalah kondisi gagal tumbuh pada anak balita (0-59 bulan) "
                "akibat kekurangan gizi kronis, terutama dalam 1000 Hari Pertama Kehidupan (HPK).\n\n"
                "Anak stunting memiliki tinggi badan di bawah standar usianya "
                "menurut kurva pertumbuhan WHO (z-score < -2 SD)."
            ),
            "penyebab_stunting": (
                "Penyebab Stunting:\n\n"
                "1. Kurangnya asupan gizi kronis (protein, vitamin, mineral)\n"
                "2. ASI tidak eksklusif 6 bulan pertama\n"
                "3. MPASI yang kurang bergizi dan tidak tepat waktu\n"
                "4. Infeksi berulang (diare, ISPA)\n"
                "5. Sanitasi lingkungan yang buruk\n"
                "6. Akses terbatas ke layanan kesehatan\n"
                "7. Pendidikan ibu yang rendah\n"
                "8. Kemiskinan dan ketahanan pangan buruk"
            ),
            "pencegahan_stunting": (
                "Cara Mencegah Stunting:\n\n"
                "1. ASI eksklusif selama 6 bulan pertama\n"
                "2. Berikan MPASI bergizi seimbang (protein hewani, sayur, buah)\n"
                "3. Penuhi gizi ibu hamil dengan tablet tambah darah\n"
                "4. Pantau pertumbuhan anak setiap bulan di posyandu\n"
                "5. Jaga kebersihan lingkungan dan sanitasi\n"
                "6. Lengkapi imunisasi dasar anak\n"
                "7. Berikan stimulasi tumbuh kembang yang tepat"
            ),
            "ciri_stunting": (
                "Ciri-ciri Anak Stunting:\n\n"
                "1. Tinggi badan di bawah standar usianya\n"
                "2. Pertumbuhan melambat (tidak naik berat/tinggi)\n"
                "3. Wajah tampak lebih muda dari usianya\n"
                "4. Keterlambatan perkembangan motorik dan kognitif\n"
                "5. Lebih pendek dari anak seusianya\n"
                "6. Mudah sakit dan daya tahan tubuh rendah"
            ),
            "makanan_stunting": (
                "Makanan untuk Cegah & Atasi Stunting:\n\n"
                "Protein Hewani (WAJIB setiap hari):\n"
                "- Telur (1 butir/hari)\n"
                "- Ikan (teri, salmon, kembung)\n"
                "- Ayam, hati ayam\n"
                "- Susu dan olahannya\n\n"
                "Karbohidrat: Nasi, ubi, jagung, kentang\n\n"
                "Sayur & Buah: Bayam, wortel, brokoli, pepaya, pisang\n\n"
                "Lemak Sehat: Minyak ikan, alpukat, santan\n\n"
                "Tips: Berikan variasi menu dan tekstur sesuai usia anak."
            ),
            "dampak_stunting": (
                "Dampak Stunting Jangka Panjang:\n\n"
                "1. Menurunnya kemampuan kognitif dan prestasi belajar\n"
                "2. Produktivitas rendah saat dewasa\n"
                "3. Risiko kemiskinan yang lebih tinggi\n"
                "4. Meningkatnya risiko penyakit tidak menular (diabetes, hipertensi)\n"
                "5. Postur tubuh pendek saat dewasa"
            ),
            "peran_orang_tua": (
                "Peran Orang Tua Mencegah Stunting:\n\n"
                "1. Memastikan asupan gizi seimbang setiap hari\n"
                "2. Memberikan ASI eksklusif dan MPASI bergizi\n"
                "3. Rutin memantau pertumbuhan ke posyandu\n"
                "4. Menjaga kebersihan lingkungan dan sanitasi\n"
                "5. Memberikan stimulasi tumbuh kembang\n"
                "6. Segera konsultasi jika ada tanda-tanda stunting"
            ),
            "unknown": (
                "Maaf, saya belum memahami pertanyaan Anda.\n\n"
                "Silakan tanyakan seputar:\n"
                "- 'Apa itu stunting?'\n"
                "- 'Kenapa anak bisa stunting?'\n"
                "- 'Bagaimana mencegah stunting?'\n"
                "- 'Makanan apa yang baik untuk cegah stunting?'\n\n"
                "Atau ketik 'bantuan' untuk panduan lengkap."
            ),
            "kapan_ke_dokter": (
                "Segera bawa anak ke dokter atau puskesmas jika:\n\n"
                "1. Tinggi badan anak tidak naik selama 2 bulan berturut-turut\n"
                "2. Berat badan turun atau stagnan lebih dari 1 bulan\n"
                "3. Anak sering sakit (diare, batuk, demam berulang)\n"
                "4. Anak tampak lemas, tidak aktif, atau nafsu makan sangat buruk\n"
                "5. Hasil deteksi aplikasi ini menunjukkan Stunting\n"
                "6. Kamu khawatir dengan tumbuh kembang anak\n\n"
                "Pemantauan rutin ke posyandu setiap bulan juga sangat dianjurkan!"
            ),
        }

    def generate(self, intent: str) -> str:
        return self.responses.get(intent, self.responses["unknown"])


class StuntingChatbot:
    def __init__(self):
        self.nlp        = SimpleNLP()
        self.classifier = IntentClassifier()
        self.generator  = ResponseGenerator()
        self.context    = ChatContext(max_history=5)

    def get_response(self, message: str) -> Dict:
        if not message or not message.strip():
            return {"response": "Mohon ketikkan pertanyaan Anda.", "intent": "empty", "confidence": 0.0}
        normalized = self.nlp.normalize(message)
        intent, confidence = self.classifier.classify(normalized, self.context)
        response_text = self.generator.generate(intent)
        self.context.add_message("user", message, intent)
        self.context.add_message("bot", response_text)
        return {"response": response_text, "intent": intent, "confidence": round(confidence, 2)}

    def reset_context(self):
        self.context.reset()

    def get_history(self) -> List[Dict]:
        return self.context.get_history()
