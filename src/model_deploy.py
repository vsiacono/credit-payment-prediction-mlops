"""
model_deploy.py

Avance 4 - Despliegue del modelo.

Expone el mejor modelo entrenado (ver model_training_evaluation.py) como un
servicio FastAPI con:
  - GET  /health          — chequeo de salud del servicio
  - POST /predict         — predicción para un solo solicitante
  - POST /predict/batch   — predicción para múltiples solicitantes (batch)

Correr localmente:
    uvicorn model_deploy:app --host 0.0.0.0 --port 8000

La imagen Docker (ver Dockerfile) empaqueta este servicio junto con el modelo
y el preprocesador ya entrenados (carpeta artifacts/).
"""

from pathlib import Path
from typing import List, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ft_engineering import CATEGORICAL_FEATURES, NUMERIC_FEATURES, ORDINAL_FEATURES, derive_features

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT / "artifacts"

app = FastAPI(
    title="API — Predicción de comportamiento de pago de créditos",
    description="Predice si un solicitante de crédito pagará a tiempo (Pago_atiempo).",
    version="1.0.0",
)

_model = None
_preprocessor = None


class CreditApplication(BaseModel):
    """Datos de un solicitante de crédito, tal como llegarían al momento del desembolso."""

    capital_prestado: float = Field(..., examples=[3000000])
    plazo_meses: int = Field(..., examples=[12])
    edad_cliente: int = Field(..., examples=[35])
    tipo_laboral: str = Field(..., examples=["Empleado"])
    salario_cliente: float = Field(..., examples=[2500000])
    total_otros_prestamos: float = Field(..., examples=[500000])
    cuota_pactada: float = Field(..., examples=[280000])
    puntaje_datacredito: Optional[float] = Field(None, examples=[720])
    cant_creditosvigentes: int = Field(..., examples=[3])
    huella_consulta: int = Field(..., examples=[2])
    saldo_mora: Optional[float] = Field(0, examples=[0])
    saldo_total: Optional[float] = Field(0, examples=[0])
    saldo_principal: Optional[float] = Field(0, examples=[0])
    saldo_mora_codeudor: Optional[float] = Field(0, examples=[0])
    creditos_sectorFinanciero: int = Field(0, examples=[1])
    creditos_sectorCooperativo: int = Field(0, examples=[0])
    creditos_sectorReal: int = Field(0, examples=[0])
    promedio_ingresos_datacredito: Optional[float] = Field(None, examples=[2500000])
    tendencia_ingresos: Optional[str] = Field(None, examples=["Estable"])
    tipo_credito: int = Field(..., examples=[4])
    fecha_prestamo: str = Field(..., examples=["2026-08-24"])


class PredictionResponse(BaseModel):
    prediccion: int
    probabilidad_pago_atiempo: float
    interpretacion: str


@app.on_event("startup")
def load_artifacts():
    global _model, _preprocessor
    try:
        _model = joblib.load(ARTIFACTS_DIR / "best_model.pkl")
        _preprocessor = joblib.load(ARTIFACTS_DIR / "preprocessor.pkl")
    except FileNotFoundError as exc:
        raise RuntimeError(
            "No se encontraron los artefactos del modelo. "
            "Corré ft_engineering.py y model_training_evaluation.py antes de levantar la API."
        ) from exc


def _applications_to_features(applications: List[CreditApplication]) -> pd.DataFrame:
    df = pd.DataFrame([a.model_dump() for a in applications])
    df["fecha_prestamo"] = pd.to_datetime(df["fecha_prestamo"])
    df = derive_features(df)
    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES + ORDINAL_FEATURES
    return df[feature_cols]


def _predict(applications: List[CreditApplication]) -> List[PredictionResponse]:
    if _model is None or _preprocessor is None:
        raise HTTPException(status_code=503, detail="El modelo no está cargado todavía.")

    X = _applications_to_features(applications)
    X_proc = _preprocessor.transform(X)
    preds = _model.predict(X_proc)
    probas = _model.predict_proba(X_proc)[:, 1]

    results = []
    for pred, proba in zip(preds, probas):
        interpretacion = (
            "Se espera que pague a tiempo." if pred == 1
            else "Riesgo de no pago a tiempo — revisar manualmente."
        )
        results.append(PredictionResponse(
            prediccion=int(pred),
            probabilidad_pago_atiempo=round(float(proba), 4),
            interpretacion=interpretacion,
        ))
    return results


@app.get("/health")
def health():
    return {"status": "ok", "modelo_cargado": _model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(application: CreditApplication):
    return _predict([application])[0]


@app.post("/predict/batch", response_model=List[PredictionResponse])
def predict_batch(applications: List[CreditApplication]):
    return _predict(applications)
