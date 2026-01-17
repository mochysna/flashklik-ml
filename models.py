# models.py
from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import ForeignKey, Text, String, Float, DateTime, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class InternalTicket(Base):
    __tablename__ = "internal_ticket"

    id_ticket: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)

    ttcheck_remark_service: Mapped[str] = mapped_column(Text, nullable=False)
    ttcheck_device_service: Mapped[str] = mapped_column(String(20), nullable=False)
    ttcheck_brand_service: Mapped[str] = mapped_column(String(100), nullable=False)
    ttcheck_series_service: Mapped[str] = mapped_column(String(150), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )

    # relasi 0..1
    report_teknisi: Mapped[Optional["InternalReportTeknisi"]] = relationship(
        back_populates="ticket",
        uselist=False,
        cascade="all, delete-orphan"
    )

    # relasi 0..1
    prediksi: Mapped[Optional["PrediksiML"]] = relationship(
        back_populates="ticket",
        uselist=False,
        cascade="all, delete-orphan"
    )


class InternalReportTeknisi(Base):
    __tablename__ = "internal_report_teknisi"

    id_report: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    id_ticket: Mapped[int] = mapped_column(ForeignKey("internal_ticket.id_ticket", ondelete="CASCADE"), unique=True)

    ttrcheck_description: Mapped[str] = mapped_column(Text, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )

    ticket: Mapped["InternalTicket"] = relationship(back_populates="report_teknisi")


class PrediksiML(Base):
    __tablename__ = "prediksi_ml"

    id_prediksi: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    id_ticket: Mapped[int] = mapped_column(ForeignKey("internal_ticket.id_ticket", ondelete="CASCADE"), unique=True)

    label_prediksi: Mapped[str] = mapped_column(String(50), nullable=False)
    confidence_score: Mapped[float] = mapped_column(Float, nullable=False)

    tanggal_prediksi: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )

    ticket: Mapped["InternalTicket"] = relationship(back_populates="prediksi")
