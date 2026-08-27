"""
ft_engineering.py

Avance 2 - Ingeniería de características.

Responsable de: limpiar la data cruda, derivar atributos adicionales, dividir en
train/evaluación, y construir el pipeline de preprocesamiento (ColumnTransformer)
que alimentará a model_training_evaluation.py.

Uso como script:
    python ft_engineering.py

Genera en ../artifacts/:
    - preprocessor.pkl      (ColumnTransformer ya ajustado sobre train)
    - X_train_raw.pkl / X_test_raw.pkl   (features sin transformar, para inspección)
    - y_train.pkl / y_test.pkl
    - X_train_proc.pkl / X_test_proc.pkl (features ya transformadas, listas para modelar)
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

TARGET = "Pago_atiempo"

# Reglas de validación / categorías válidas para tendencia_ingresos (ver comprension_eda.ipynb)
CATEGORIAS_TENDENCIA = ["Decreciente", "Estable", "Creciente"]

NUMERIC_FEATURES = [
    "capital_prestado",
    "plazo_meses",
    "edad_cliente",
    "salario_cliente",
    "total_otros_prestamos",
    "cuota_pactada",
    "puntaje_datacredito",
    "cant_creditosvigentes",
    "huella_consulta",
    "saldo_mora",
    "saldo_total",
    "saldo_principal",
    "saldo_mora_codeudor",
    "creditos_sectorFinanciero",
    "creditos_sectorCooperativo",
    "creditos_sectorReal",
    "promedio_ingresos_datacredito",
    # derivadas (ver derive_features)
    "relacion_cuota_salario",
    "relacion_deuda_salario",
    "saldo_mora_total",
    "total_creditos_sector",
    "mes_prestamo",
]

# NOTA: se excluye intencionalmente "puntaje" del set de features. Al revisarlo
# se detectó que separa casi perfectamente la variable objetivo (ver EDA), lo cual
# indica que es un score calculado DESPUÉS de observar el comportamiento de pago
# (data leakage) y no estaría disponible al momento de evaluar un cliente nuevo.
# "puntaje_datacredito" sí se conserva: es un score de central de riesgo, externo
# y disponible antes del desembolso del crédito.

CATEGORICAL_FEATURES = ["tipo_credito", "tipo_laboral"]

# tendencia_ingresos se trata como ordinal: hay un orden natural
# Decreciente < Estable < Creciente
ORDINAL_FEATURES = ["tendencia_ingresos"]


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica las reglas de validación identificadas en comprension_eda.ipynb."""
    df = df.copy()

    # Regla: tendencia_ingresos solo admite Estable/Creciente/Decreciente o nulo.
    # Cualquier otro valor (numérico colado) se convierte a NaN.
    df["tendencia_ingresos"] = df["tendencia_ingresos"].where(
        df["tendencia_ingresos"].isin(CATEGORIAS_TENDENCIA)
    )

    # Regla: salario_cliente y total_otros_prestamos con valores extremos
    # (varios órdenes de magnitud por encima de lo razonable) se topean al
    # percentil 99 en vez de descartar la fila (evita perder información del resto
    # de las variables de esos clientes).
    for col in ["salario_cliente", "total_otros_prestamos"]:
        p99 = df[col].quantile(0.99)
        df[col] = df[col].clip(upper=p99)

    df["fecha_prestamo"] = pd.to_datetime(df["fecha_prestamo"])

    return df


def derive_features(df: pd.DataFrame) -> pd.DataFrame:
    """Crea atributos adicionales identificados en el EDA."""
    df = df.copy()

    df["relacion_cuota_salario"] = df["cuota_pactada"] / df["salario_cliente"].replace(0, np.nan)
    df["relacion_deuda_salario"] = df["total_otros_prestamos"] / df["salario_cliente"].replace(0, np.nan)
    df["saldo_mora_total"] = df["saldo_mora"].fillna(0) + df["saldo_mora_codeudor"].fillna(0)
    df["total_creditos_sector"] = (
        df["creditos_sectorFinanciero"] + df["creditos_sectorCooperativo"] + df["creditos_sectorReal"]
    )
    df["mes_prestamo"] = df["fecha_prestamo"].dt.month

    return df


def build_preprocessor() -> ColumnTransformer:
    """Pipeline numeric / categoric / categoric ordinal, según el diagrama del avance."""
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    ordinal_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ordinal", OrdinalEncoder(categories=[CATEGORIAS_TENDENCIA])),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("numeric", numeric_transformer, NUMERIC_FEATURES),
        ("categoric", categorical_transformer, CATEGORICAL_FEATURES),
        ("categoric_ordinal", ordinal_transformer, ORDINAL_FEATURES),
    ])

    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> list:
    return list(preprocessor.get_feature_names_out())


def get_train_test(df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
    """Limpia, deriva features y devuelve el split train/evaluación (sin transformar)."""
    df = clean_data(df)
    df = derive_features(df)

    feature_cols = NUMERIC_FEATURES + CATEGORICAL_FEATURES + ORDINAL_FEATURES
    X = df[feature_cols]
    y = df[TARGET]

    # stratify=y porque Pago_atiempo está fuertemente desbalanceada (~95/5)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


def main():
    root = Path(__file__).resolve().parent.parent
    data_path = root / "Base_de_datos.csv"
    artifacts_dir = root / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)

    df = pd.read_csv(data_path)
    X_train, X_test, y_train, y_test = get_train_test(df)

    preprocessor = build_preprocessor()
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    joblib.dump(preprocessor, artifacts_dir / "preprocessor.pkl")
    joblib.dump(X_train, artifacts_dir / "X_train_raw.pkl")
    joblib.dump(X_test, artifacts_dir / "X_test_raw.pkl")
    joblib.dump(y_train, artifacts_dir / "y_train.pkl")
    joblib.dump(y_test, artifacts_dir / "y_test.pkl")
    joblib.dump(X_train_proc, artifacts_dir / "X_train_proc.pkl")
    joblib.dump(X_test_proc, artifacts_dir / "X_test_proc.pkl")
    joblib.dump(get_feature_names(preprocessor), artifacts_dir / "feature_names.pkl")

    print(f"Train: {X_train.shape}  |  Test: {X_test.shape}")
    print(f"Features tras preprocesamiento: {X_train_proc.shape[1]}")
    print(f"Artefactos guardados en {artifacts_dir}")


if __name__ == "__main__":
    main()
