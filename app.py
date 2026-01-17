from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

# Load model (pastikan file ini ada di folder yang sama saat run uvicorn)
model = joblib.load("artifacts/model_keluhan_decision_tree.joblib")

app = FastAPI(title="API Prediksi Keluhan FlashKlik (Decision Tree)")

# --- Mapping rekomendasi teknisi per label ---
REKOMENDASI = {
    "POWER": "Cek adaptor/charger, port DC, jalur power. Tes dengan charger lain bila tersedia.",
    "BATTERY": "Cek kesehatan baterai, konektor baterai, dan arus charging. Lakukan uji dengan adaptor normal.",
    "MOTHERBOARD": "Cek indikasi short/komponen panas, uji arus/tegangan, inspeksi visual untuk korosi/bekas terbakar.",
    "OVERHEAT": "Bersihkan fan/heatsink, cek thermal paste, lakukan stress test suhu setelah tindakan.",
    "KEYBOARD": "Uji semua tombol, cek fleksibel/connector keyboard, coba external keyboard untuk pembanding.",
    "TOUCHPAD": "Cek driver, uji fungsi click/gesture, cek fleksibel/connector touchpad.",
    "ENGSEL": "Cek kondisi engsel & housing, pastikan tidak merusak frame/LCD cable, sarankan penguatan/penggantian part.",
    "DISPLAY": "Tes dengan external monitor, cek kabel fleksibel LCD, kondisi panel/backlight, dan konektor.",
    "PRINTER": "Cek paper jam/path kertas, cleaning head/nozzle check, cek cartridge/tinta dan roller.",
    "STORAGE": "Cek SMART/health SSD-HDD, uji bad sector, pastikan konektor/storage slot normal.",
    "SOFTWARE": "Cek driver/OS, lakukan repair/reinstall bila perlu, scan malware, cek update firmware/BIOS.",
    "LAINNYA": "Perlu pemeriksaan lanjutan untuk diagnosis yang lebih spesifik (cek part terkait sesuai gejala).",
}

# --- Request/Response Schema ---
class PredictRequest(BaseModel):
    jenis_perangkat: str
    merk: str
    series: str | None = ""
    keluhan: str

class TopKItem(BaseModel):
    label: str
    persen: float
    confidence: float

class PredictResponse(BaseModel):
    perkiraan_kerusakan: str
    confidence: float
    confidence_persen: float
    rekomendasi: str
    pesan_teknisi: str
    top_3_kemungkinan: list[TopKItem]

def _norm(s: str | None) -> str:
    return (s or "").strip()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    # Normalisasi input
    jenis = _norm(req.jenis_perangkat).upper()
    merk = _norm(req.merk).upper()
    seri = _norm(req.series).upper()
    keluhan = _norm(req.keluhan)

    # WAJIB: kolom text_combined harus ada karena training pakai text_col="text_combined"
    # Bisa sederhana (hanya keluhan), atau gabungkan biar lebih kaya informasi:
    text_combined = keluhan
    # text_combined = f"{keluhan} {jenis} {merk} {seri}".strip()

    # Input harus DataFrame dengan nama kolom sama seperti training
    X_input = pd.DataFrame([{
        "text_combined": text_combined,
        "ttcheck_device_service": jenis,
        "ttcheck_brand_service": merk,
        "ttcheck_series_service": seri
    }])

    # Prediksi probabilitas (DecisionTreeClassifier support predict_proba)
    proba = model.predict_proba(X_input)[0]
    classes = model.classes_

    # Ambil top-3
    top_k = min(3, len(classes))
    idx_sorted = np.argsort(proba)[::-1][:top_k]

    top3: list[dict] = []
    for i in idx_sorted:
        conf = float(proba[i])
        top3.append({
            "label": str(classes[i]),
            "confidence": conf,
            "persen": round(conf * 100, 2)
        })

    # Prediksi utama
    best = top3[0]
    label = best["label"]
    conf = best["confidence"]

    rekom = REKOMENDASI.get(label, REKOMENDASI["LAINNYA"])
    pesan = f"Perkiraan kerusakan utama: {label} ({conf*100:.1f}%). Saran awal: {rekom}"

    return PredictResponse(
        perkiraan_kerusakan=label,
        confidence=conf,
        confidence_persen=round(conf * 100, 2),
        rekomendasi=rekom,
        pesan_teknisi=pesan,
        top_3_kemungkinan=top3
    )
