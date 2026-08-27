"""
model_monitoring.py

Avance 3 - Monitoreo y detección de data drift.

Contiene:
  1) Funciones de cálculo de métricas de drift (KS test, PSI, Jensen-Shannon,
     Chi-cuadrado) comparando una distribución de referencia (datos con los que
     se entrenó el modelo) contra una distribución "actual" (datos nuevos /
     muestreo periódico).
  2) Una app de Streamlit que visualiza esas métricas: tabla con semáforo de
     alerta, comparación de distribuciones, evolución temporal y recomendaciones.

Uso como job de monitoreo (sin UI):
    python model_monitoring.py

Uso como app:
    streamlit run model_monitoring.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import chi2_contingency, ks_2samp

NUMERIC_DRIFT_COLS = [
    "capital_prestado",
    "plazo_meses",
    "edad_cliente",
    "salario_cliente",
    "total_otros_prestamos",
    "cuota_pactada",
    "puntaje_datacredito",
    "cant_creditosvigentes",
    "huella_consulta",
    "saldo_total",
]
CATEGORICAL_DRIFT_COLS = ["tipo_credito", "tipo_laboral", "tendencia_ingresos"]

# Umbrales de alerta (referencias estándar de industria para riesgo crediticio)
PSI_WARN, PSI_ALERT = 0.1, 0.25
KS_PVALUE_ALERT = 0.05
JS_ALERT = 0.1
CHI2_PVALUE_ALERT = 0.05


# --------------------------------------------------------------------------
# Métricas de drift
# --------------------------------------------------------------------------

def compute_ks(reference: pd.Series, current: pd.Series) -> tuple:
    """Kolmogorov-Smirnov test. Devuelve (estadístico, p-value)."""
    ref = reference.dropna()
    cur = current.dropna()
    stat, pvalue = ks_2samp(ref, cur)
    return stat, pvalue


def compute_psi(reference: pd.Series, current: pd.Series, buckets: int = 10) -> float:
    """Population Stability Index sobre una variable numérica, en deciles de referencia."""
    ref = reference.dropna()
    cur = current.dropna()

    quantiles = np.linspace(0, 1, buckets + 1)
    breakpoints = np.unique(ref.quantile(quantiles).values)
    if len(breakpoints) < 3:
        return 0.0  # variable casi constante, no hay drift medible por bucket

    ref_counts, _ = np.histogram(ref, bins=breakpoints)
    cur_counts, _ = np.histogram(cur, bins=breakpoints)

    ref_pct = np.where(ref_counts == 0, 1e-4, ref_counts / ref_counts.sum())
    cur_pct = np.where(cur_counts == 0, 1e-4, cur_counts / cur_counts.sum())

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


def compute_js_divergence(reference: pd.Series, current: pd.Series, buckets: int = 10) -> float:
    """Jensen-Shannon divergence sobre histogramas normalizados de una variable numérica."""
    ref = reference.dropna()
    cur = current.dropna()

    breakpoints = np.histogram_bin_edges(pd.concat([ref, cur]), bins=buckets)
    ref_counts, _ = np.histogram(ref, bins=breakpoints)
    cur_counts, _ = np.histogram(cur, bins=breakpoints)

    ref_pct = ref_counts / max(ref_counts.sum(), 1)
    cur_pct = cur_counts / max(cur_counts.sum(), 1)

    return float(jensenshannon(ref_pct, cur_pct, base=2))


def compute_chi2(reference: pd.Series, current: pd.Series) -> tuple:
    """Chi-cuadrado de independencia para una variable categórica. Devuelve (estadístico, p-value)."""
    ref_counts = reference.value_counts()
    cur_counts = current.value_counts()
    categories = sorted(set(ref_counts.index) | set(cur_counts.index))

    contingency = pd.DataFrame({
        "reference": [ref_counts.get(c, 0) for c in categories],
        "current": [cur_counts.get(c, 0) for c in categories],
    }, index=categories)

    stat, pvalue, _, _ = chi2_contingency(contingency)
    return stat, pvalue


def generate_drift_report(reference: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    """Reporte consolidado de drift por variable, con flag de alerta."""
    rows = []

    for col in NUMERIC_DRIFT_COLS:
        ks_stat, ks_p = compute_ks(reference[col], current[col])
        psi = compute_psi(reference[col], current[col])
        js = compute_js_divergence(reference[col], current[col])
        alerta = (psi >= PSI_ALERT) or (ks_p < KS_PVALUE_ALERT) or (js >= JS_ALERT)
        rows.append({
            "variable": col, "tipo": "numerica",
            "ks_stat": round(ks_stat, 4), "ks_pvalue": round(ks_p, 4),
            "psi": round(psi, 4), "js_divergence": round(js, 4),
            "chi2_pvalue": np.nan, "alerta_drift": alerta,
        })

    for col in CATEGORICAL_DRIFT_COLS:
        chi2_stat, chi2_p = compute_chi2(reference[col].astype(str), current[col].astype(str))
        alerta = chi2_p < CHI2_PVALUE_ALERT
        rows.append({
            "variable": col, "tipo": "categorica",
            "ks_stat": np.nan, "ks_pvalue": np.nan,
            "psi": np.nan, "js_divergence": np.nan,
            "chi2_pvalue": round(chi2_p, 4), "alerta_drift": alerta,
        })

    return pd.DataFrame(rows).sort_values("alerta_drift", ascending=False).reset_index(drop=True)


def sample_periodic(df: pd.DataFrame, date_col: str, n_periods: int = 6) -> pd.DataFrame:
    """Muestreo periódico: asigna cada fila a un período (mes) para análisis temporal."""
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["periodo"] = df[date_col].dt.to_period("M").astype(str)
    return df


def drift_over_time(df: pd.DataFrame, date_col: str, variable: str) -> pd.DataFrame:
    """PSI de una variable en cada período respecto del primer período (referencia)."""
    df = sample_periodic(df, date_col)
    periodos = sorted(df["periodo"].unique())
    if len(periodos) < 2:
        return pd.DataFrame(columns=["periodo", "psi"])

    reference = df[df["periodo"] == periodos[0]][variable]
    rows = [{"periodo": periodos[0], "psi": 0.0}]
    for p in periodos[1:]:
        current = df[df["periodo"] == p][variable]
        rows.append({"periodo": p, "psi": compute_psi(reference, current)})
    return pd.DataFrame(rows)


def run_monitoring_job(reference: pd.DataFrame, current: pd.DataFrame) -> pd.DataFrame:
    """Job de monitoreo: genera y devuelve el reporte de drift (para uso periódico/batch)."""
    report = generate_drift_report(reference, current)
    n_alertas = report["alerta_drift"].sum()
    print(f"Variables con drift detectado: {n_alertas} / {len(report)}")
    return report


# --------------------------------------------------------------------------
# Job batch (sin UI) — separa el dataset en dos mitades por fecha para simular
# "referencia" (con la que se entrenó) vs "actual" (créditos más recientes)
# --------------------------------------------------------------------------

def _load_reference_current():
    root = Path(__file__).resolve().parent.parent
    df = pd.read_csv(root / "Base_de_datos.csv")
    df["fecha_prestamo"] = pd.to_datetime(df["fecha_prestamo"])
    df = df.sort_values("fecha_prestamo")
    mid = len(df) // 2
    reference, current = df.iloc[:mid], df.iloc[mid:]
    return df, reference, current


def main():
    root = Path(__file__).resolve().parent.parent
    artifacts_dir = root / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)

    _, reference, current = _load_reference_current()
    report = run_monitoring_job(reference, current)
    report.to_csv(artifacts_dir / "drift_report.csv", index=False)
    print(f"Reporte guardado en {artifacts_dir / 'drift_report.csv'}")


# --------------------------------------------------------------------------
# App de Streamlit — se ejecuta con: streamlit run model_monitoring.py
# --------------------------------------------------------------------------

def _run_streamlit_app():
    import streamlit as st

    st.set_page_config(page_title="Monitoreo de modelo — Créditos", layout="wide")
    st.title("📊 Monitoreo de modelo — Comportamiento de pago de créditos")
    st.caption(
        "Caso de negocio: detectar cambios (data drift) en el perfil de los "
        "solicitantes de crédito respecto a los datos con los que se entrenó "
        "el modelo, para anticipar degradación de performance antes de que "
        "impacte el negocio."
    )

    df, reference, current = _load_reference_current()
    report = generate_drift_report(reference, current)

    n_alertas = int(report["alerta_drift"].sum())
    col1, col2, col3 = st.columns(3)
    col1.metric("Variables monitoreadas", len(report))
    col2.metric("Con alerta de drift", n_alertas)
    col3.metric(
        "Estado general",
        "🔴 Revisar" if n_alertas > len(report) * 0.3 else ("🟡 Vigilar" if n_alertas > 0 else "🟢 OK"),
    )

    st.subheader("Tabla de métricas de drift por variable")

    def semaforo(row):
        if not row["alerta_drift"]:
            return "🟢"
        return "🔴"

    report_display = report.copy()
    report_display.insert(0, "estado", report.apply(semaforo, axis=1))
    st.dataframe(report_display, use_container_width=True)

    st.subheader("Comparación de distribuciones: referencia vs actual")
    variable = st.selectbox("Variable", NUMERIC_DRIFT_COLS + CATEGORICAL_DRIFT_COLS)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    if variable in NUMERIC_DRIFT_COLS:
        ax.hist(reference[variable].dropna(), bins=30, alpha=0.5, label="Referencia (train)", density=True)
        ax.hist(current[variable].dropna(), bins=30, alpha=0.5, label="Actual", density=True)
    else:
        ref_pct = reference[variable].astype(str).value_counts(normalize=True)
        cur_pct = current[variable].astype(str).value_counts(normalize=True)
        comp = pd.DataFrame({"Referencia": ref_pct, "Actual": cur_pct}).fillna(0)
        comp.plot(kind="bar", ax=ax)
    ax.set_title(f"Distribución de {variable}")
    ax.legend()
    st.pyplot(fig)

    st.subheader("Evolución temporal del drift")
    variable_temporal = st.selectbox("Variable para evolución temporal", NUMERIC_DRIFT_COLS, key="temporal")
    evolucion = drift_over_time(df, "fecha_prestamo", variable_temporal)
    st.line_chart(evolucion.set_index("periodo")["psi"])
    st.caption("PSI por mes respecto al primer período disponible. PSI > 0.25 sugiere cambio significativo.")

    st.subheader("Recomendaciones")
    if n_alertas == 0:
        st.success("No se detectó drift significativo. No se requiere acción inmediata.")
    else:
        variables_alerta = report[report["alerta_drift"]]["variable"].tolist()
        st.warning(
            f"Se detectó drift en: {', '.join(variables_alerta)}. "
            "Se recomienda revisar el pipeline de ingesta de esas variables, "
            "evaluar re-entrenamiento del modelo con datos recientes, y "
            "monitorear de cerca la performance real (accuracy/recall) en producción."
        )


if __name__ == "__main__":
    import sys

    # Si se ejecuta con `streamlit run`, streamlit ya está en sys.modules antes
    # de correr este script; en ese caso mostramos la app en vez del job batch.
    if "streamlit" in sys.modules or any("streamlit" in arg for arg in sys.argv):
        _run_streamlit_app()
    else:
        main()
