"""
model_training_evaluation.py

Avance 2 - Entrenamiento y evaluación de modelos supervisados.

Entrena varios modelos de clasificación sobre los artefactos generados por
ft_engineering.py, los compara con métricas apropiadas para un problema
desbalanceado (Pago_atiempo ~95/5) y selecciona el de mejor performance.

Uso como script:
    python model_training_evaluation.py

Genera en ../artifacts/:
    - metrics_comparison.csv
    - comparison_charts.png
    - best_model.pkl
"""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier


def build_model(model, X_train, y_train):
    """Entrena (fit) un modelo ya instanciado y lo devuelve."""
    model.fit(X_train, y_train)
    return model


def summarize_classification(model, X_test, y_test, model_name: str) -> dict:
    """Calcula el set de métricas estándar para un clasificador binario."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return {
        "modelo": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }


def get_models(count_neg: int, count_pos: int) -> dict:
    """Modelos candidatos. class_weight/scale_pos_weight por el fuerte desbalanceo."""
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced_subsample", random_state=42, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            eval_metric="logloss",
            random_state=42,
            # La clase minoritaria es Pago_atiempo=0 (no pagó a tiempo, ~5%),
            # no la clase 1: scale_pos_weight = negativos/positivos = count(0)/count(1)
            scale_pos_weight=count_neg / count_pos,
        ),
    }


def main():
    root = Path(__file__).resolve().parent.parent
    artifacts_dir = root / "artifacts"

    X_train = joblib.load(artifacts_dir / "X_train_proc.pkl")
    X_test = joblib.load(artifacts_dir / "X_test_proc.pkl")
    y_train = joblib.load(artifacts_dir / "y_train.pkl")
    y_test = joblib.load(artifacts_dir / "y_test.pkl")

    count_neg = int((y_train == 0).sum())
    count_pos = int((y_train == 1).sum())
    models = get_models(count_neg, count_pos)
    trained = {}
    results = []

    for name, model in models.items():
        print(f"Entrenando {name}...")
        trained[name] = build_model(model, X_train, y_train)
        results.append(summarize_classification(trained[name], X_test, y_test, name))

    metrics_df = pd.DataFrame(results).sort_values("roc_auc", ascending=False).reset_index(drop=True)
    print("\nTabla resumen de evaluación:")
    print(metrics_df.to_string(index=False))

    metrics_df.to_csv(artifacts_dir / "metrics_comparison.csv", index=False)

    # --- Gráficos comparativos ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    metrics_melt = metrics_df.set_index("modelo")[["precision", "recall", "f1", "roc_auc"]]
    metrics_melt.plot(kind="bar", ax=axes[0])
    axes[0].set_title("Comparación de métricas por modelo")
    axes[0].set_ylabel("score")
    axes[0].tick_params(axis="x", rotation=20)

    for name, model in trained.items():
        RocCurveDisplay.from_estimator(model, X_test, y_test, ax=axes[1], name=name)
    axes[1].set_title("Curvas ROC")

    best_name = metrics_df.iloc[0]["modelo"]
    best_model = trained[best_name]
    ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test, ax=axes[2], colorbar=False)
    axes[2].set_title(f"Matriz de confusión — mejor modelo ({best_name})")

    plt.tight_layout()
    plt.savefig(artifacts_dir / "comparison_charts.png", dpi=120)
    plt.close()

    joblib.dump(best_model, artifacts_dir / "best_model.pkl")
    print(f"\nMejor modelo por ROC-AUC: {best_name} → guardado en best_model.pkl")


if __name__ == "__main__":
    main()
