from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

from fastapi import Depends, HTTPException
from sqlalchemy import text

from sqlalchemy.orm import Session
from db import get_db, init_db
from models import InternalTicket, InternalReportTeknisi, PrediksiML

# Load model (pastikan file ini ada di folder yang sama saat run uvicorn)
model = joblib.load("artifacts/model_keluhan_decision_tree.joblib")

app = FastAPI(title="API Prediksi Keluhan FlashKlik (Decision Tree)")

@app.on_event("startup")
def on_startup():
    init_db()

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

@app.get("/db-health")
def db_health(db: Session = Depends(get_db)):
    row = db.execute(text("SELECT current_database() AS db, version() AS version")).mappings().first()
    return {
        "status": "ok",
        "database": row["db"],
        "postgres_version": row["version"],
    }

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    # Normalisasi input
    jenis = _norm(req.jenis_perangkat).lower()
    merk = _norm(req.merk).lower()
    seri = _norm(req.series).lower()
    keluhan = _norm(req.keluhan)

    # WAJIB: kolom text_combined harus ada karena training pakai text_col="text_combined"
    text_combined = keluhan
    # text_combined = f"{keluhan} {jenis} {merk} {seri}".strip()

    # Input harus DataFrame dengan nama kolom sama seperti training
    X_input = pd.DataFrame([{
        "text_combined": text_combined,
        "ttcheck_device_service": jenis,
        "ttcheck_brand_service": merk,
        "ttcheck_series_service": seri
    }])

    # Prediksi probabilitas
    proba = model.predict_proba(X_input)[0]
    classes = model.classes_

    # --- TOP-K FILTERED ---
    EPS = 1e-6
    TOP_K = min(3, len(classes))

    idx_sorted = np.argsort(proba)[::-1]

    top3: list[dict] = []
    for i in idx_sorted[:TOP_K]:
        conf_i = float(proba[i])
        if conf_i > EPS:
            top3.append({
                "label": str(classes[i]),
                "confidence": conf_i,
                "persen": round(conf_i * 100, 2)
            })

    # fallback: kalau semua conf dianggap 0
    if not top3:
        i0 = int(idx_sorted[0])
        conf0 = float(proba[i0])
        top3 = [{
            "label": str(classes[i0]),
            "confidence": conf0,
            "persen": round(conf0 * 100, 2)
        }]

    # kalau prediksi utama ~100%, tampilkan 1 saja
    if abs(top3[0]["confidence"] - 1.0) < EPS:
        top3 = [top3[0]]
    # --- END TOP-K FILTERED ---

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


class TicketCreate(BaseModel):
    ttcheck_remark_service: str
    ttcheck_device_service: str
    ttcheck_brand_service: str
    ttcheck_series_service: str | None = ""

@app.post("/tickets")
def create_ticket(payload: TicketCreate, db: Session = Depends(get_db)):
    t = InternalTicket(
        ttcheck_remark_service=payload.ttcheck_remark_service.strip(),
        ttcheck_device_service=payload.ttcheck_device_service.strip().upper(),
        ttcheck_brand_service=payload.ttcheck_brand_service.strip().upper(),
        ttcheck_series_service=(payload.ttcheck_series_service or "").strip().upper(),
    )
    db.add(t)
    db.commit()
    db.refresh(t)
    return {"id_ticket": t.id_ticket}

class ReportCreate(BaseModel):
    ttrcheck_description: str

@app.post("/tickets/{id_ticket}/report")
def upsert_report(id_ticket: int, payload: ReportCreate, db: Session = Depends(get_db)):
    ticket = db.get(InternalTicket, id_ticket)
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket tidak ditemukan")

    existing = db.query(InternalReportTeknisi).filter_by(id_ticket=id_ticket).first()
    if existing:
        existing.ttrcheck_description = payload.ttrcheck_description.strip()
        db.commit()
        return {"status": "updated"}
    else:
        r = InternalReportTeknisi(id_ticket=id_ticket, ttrcheck_description=payload.ttrcheck_description.strip())
        db.add(r)
        db.commit()
        return {"status": "created"}

@app.post("/tickets/{id_ticket}/predict")
def predict_for_ticket(id_ticket: int, db: Session = Depends(get_db)):
    ticket = db.get(InternalTicket, id_ticket)
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket tidak ditemukan")

    report = db.query(InternalReportTeknisi).filter_by(id_ticket=id_ticket).first()

    # text_combined: gabung remark + report teknisi (kalau ada)
    keluhan = (ticket.ttcheck_remark_service or "").strip()
    teknisi = (report.ttrcheck_description.strip() if report else "")
    text_combined = (keluhan + " " + teknisi).strip()

    X_input = pd.DataFrame([{
        "text_combined": text_combined,
        "ttcheck_device_service": (ticket.ttcheck_device_service or "").strip().upper(),
        "ttcheck_brand_service": (ticket.ttcheck_brand_service or "").strip().upper(),
        "ttcheck_series_service": (ticket.ttcheck_series_service or "").strip().upper(),
    }])

    proba = model.predict_proba(X_input)[0]
    classes = model.classes_

    # ambil top yang >0, dan kalau 100% ya tampil 1 aja
    EPS = 1e-6
    idx_sorted = np.argsort(proba)[::-1]

    top = []
    for i in idx_sorted[:min(3, len(classes))]:
        conf = float(proba[i])
        if conf > EPS:
            top.append({"label": str(classes[i]), "confidence": conf, "persen": round(conf*100, 2)})

    if not top:
        i0 = int(idx_sorted[0])
        top = [{"label": str(classes[i0]), "confidence": float(proba[i0]), "persen": round(float(proba[i0])*100, 2)}]

    if abs(top[0]["confidence"] - 1.0) < EPS:
        top = [top[0]]

    label = top[0]["label"]
    conf = top[0]["confidence"]

    # upsert prediksi_ml (0..1 per ticket)
    # upsert prediksi_ml (0..1 per ticket)
    pred = db.query(PrediksiML).filter_by(id_ticket=id_ticket).first()
    if pred:
        pred.label_prediksi = label
        pred.confidence_score = conf
    else:
        pred = PrediksiML(
            id_ticket=id_ticket,
            label_prediksi=label,
            confidence_score=conf,
        )
        db.add(pred)

    db.commit()
    return {
        "id_ticket": id_ticket,
        "perkiraan_kerusakan": label,
        "confidence": conf,
        "top_kemungkinan": top
    }

@app.get("/tickets/{id_ticket}")
def get_ticket(id_ticket: int, db: Session = Depends(get_db)):
    ticket = db.get(InternalTicket, id_ticket)
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket tidak ditemukan")

    report = db.query(InternalReportTeknisi).filter_by(id_ticket=id_ticket).first()
    pred = db.query(PrediksiML).filter_by(id_ticket=id_ticket).first()

    return {
        "ticket": {
            "id_ticket": ticket.id_ticket,
            "remark": ticket.ttcheck_remark_service,
            "device": ticket.ttcheck_device_service,
            "brand": ticket.ttcheck_brand_service,
            "series": ticket.ttcheck_series_service,
        },
        "report_teknisi": (report.ttrcheck_description if report else None),
        "prediksi": ({
            "label": pred.label_prediksi,
            "confidence": pred.confidence_score,
            # "topk": pred.raw_topk
        } if pred else None)
    }
