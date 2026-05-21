# 🩺 Stunting Prediction API

FastAPI service untuk prediksi status stunting pada anak menggunakan model XGBoost.

---

## 📁 Struktur Proyek

```
stunting_api/
├── app/
│   ├── main.py              # Aplikasi FastAPI
│   ├── model_stunting.pkl   # Model XGBoost
│   └── scaler.pkl           # StandardScaler
├── requirements.txt
└── README.md
```

---

## ⚙️ Instalasi

```bash
# 1. Clone / copy proyek ini
cd stunting_api

# 2. Buat virtual environment (opsional tapi dianjurkan)
python -m venv venv
source venv/bin/activate        # Linux/Mac
# venv\Scripts\activate         # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Menjalankan Server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Server akan berjalan di: `http://localhost:8000`

---

## 📖 Dokumentasi

| URL | Keterangan |
|-----|------------|
| `http://localhost:8000/docs` | Swagger UI (interaktif) |
| `http://localhost:8000/redoc` | ReDoc |

---

## 🔌 Endpoint

### `GET /`
Root endpoint, cek API aktif.

### `GET /health`
Cek status model & scaler.

### `POST /predict`
Prediksi status stunting.

**Request Body:**
```json
{
  "age": 24,
  "height": 82.5,
  "weight": 11.2,
  "bmi": 16.4
}
```

**Response:**
```json
{
  "prediction": 1,
  "label": "Stunting",
  "probability_normal": 0.1823,
  "probability_stunting": 0.8177,
  "input": {
    "age": 24,
    "height": 82.5,
    "weight": 11.2,
    "bmi": 16.4
  }
}
```

**Label:**
- `0` → Normal
- `1` → Stunting

---

## 🧪 Contoh cURL

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"age": 24, "height": 82.5, "weight": 11.2, "bmi": 16.4}'
```
