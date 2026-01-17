from sqlalchemy import (
    BigInteger, Text, String, Float, DateTime, ForeignKey, JSON, func
)
from sqlalchemy.orm import Mapped, mapped_column, relationship
from db import Base

class InternalTicket(Base):
    __tablename__ = "internal_ticket"

    id_ticket: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    ttcheck_remark_service: Mapped[str] = mapped_column(Text, nullable=False)
    ttcheck_device_service: Mapped[str] = mapped_column(String(20), nullable=False)
    ttcheck_brand_service: Mapped[str] = mapped_column(String(80), nullable=False)
    ttcheck_series_service: Mapped[str] = mapped_column(String(120), default="", nullable=True)

    created_at: Mapped = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    report = relationship("InternalReportTeknisi", back_populates="ticket", uselist=False, cascade="all, delete")
    prediksi = relationship("PrediksiML", back_populates="ticket", uselist=False, cascade="all, delete")


class InternalReportTeknisi(Base):
    __tablename__ = "internal_report_teknisi"

    id_report: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    id_ticket: Mapped[int] = mapped_column(BigInteger, ForeignKey("internal_ticket.id_ticket", ondelete="CASCADE"), unique=True, nullable=False)
    ttrcheck_description: Mapped[str] = mapped_column(Text, nullable=False)

    created_at: Mapped = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at: Mapped = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    ticket = relationship("InternalTicket", back_populates="report")


class PrediksiML(Base):
    __tablename__ = "prediksi_ml"

    id_prediksi: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    id_ticket: Mapped[int] = mapped_column(BigInteger, ForeignKey("internal_ticket.id_ticket", ondelete="CASCADE"), unique=True, nullable=False)

    label_prediksi: Mapped[str] = mapped_column(String(40), nullable=False)
    confidence_score: Mapped[float] = mapped_column(Float, nullable=False)
    tanggal_prediksi: Mapped = mapped_column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    raw_topk: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    ticket = relationship("InternalTicket", back_populates="prediksi")
