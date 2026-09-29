# Credit Payment Prediction — MLOps Pipeline

Proyecto de Machine Learning y MLOps orientado a predecir si un solicitante de crédito pagará a tiempo (`Pago_atiempo`) a partir de información histórica de créditos.

El proyecto cubre el ciclo de vida completo de un modelo de Machine Learning: análisis exploratorio de datos, ingeniería de características, entrenamiento y evaluación de modelos, monitoreo de data drift y despliegue mediante una API.

## Objetivo del proyecto

Desarrollar un pipeline reproducible para analizar el comportamiento de pago de créditos y llevar un modelo desde la exploración de datos hasta su monitoreo y despliegue.

El dataset contiene **10.763 registros históricos y 23 variables**. Durante el análisis se identificó y excluyó la variable `puntaje` por presentar **data leakage**, evitando utilizar información que no estaría disponible al evaluar un nuevo solicitante.

## Pipeline

**EDA → Feature Engineering → Model Training → Evaluation → Data Drift Monitoring → API Deployment**

Se compararon modelos de **Regresión Logística, Random Forest y XGBoost**, considerando el fuerte desbalanceo de la variable objetivo. La selección del modelo se realizó mediante **ROC-AUC**.

El proyecto también incorpora monitoreo de cambios en las distribuciones de los datos mediante **KS Test, PSI, Jensen-Shannon Divergence y Chi-cuadrado**, una aplicación de monitoreo en **Streamlit** y una API desarrollada con **FastAPI** y preparada para contenerización con **Docker**.

## Tecnologías

`Python` · `Pandas` · `NumPy` · `Scikit-learn` · `XGBoost` · `SciPy` · `Matplotlib` · `Streamlit` · `FastAPI` · `Docker` · `Jupyter` · `Git/GitHub`

## Estructura del repositorio

```text
credit-payment-prediction-mlops/
├── src/
│   ├── config.json
│   ├── Cargar_datos.ipynb
│   ├── comprension_eda.ipynb
│   ├── ft_engineering.py
│   ├── model_training_evaluation.py
│   ├── model_monitoring.py
│   └── model_deploy.py
├── Base_de_datos.csv
├── Dockerfile
├── .dockerignore
├── requirements.txt
├── .gitignore
├── set_up.bat
└── LICENSE

## Ramas

El repositorio usa un flujo de tres ramas:

- **developer** — desarrollo activo de notebooks y scripts.
- **certification** — rama de control/QA intermedia.
- **master** — versión estable, integrada vía Pull Request aprobado por un compañero.

## Configuración del entorno

1. Cloná el repositorio.
2. Ejecutá `set_up.bat` desde la raíz del proyecto (Windows). El script:
   - Lee `project_code` desde `src/config.json`.
   - Crea el entorno virtual `<project_code>-venv`.
   - Instala las dependencias de `requirements.txt`.
   - Registra el entorno como kernel de Jupyter.
3. Abrí los notebooks en `src/` seleccionando el kernel `<project_code>-venv Python ETL`.

## Avance 1 — Versionamiento y Colaboración

- `Cargar_datos.ipynb`: carga del dataset de ejemplo (`Base_de_datos.csv`, dataset
  no productivo — en producción los datos provendrían del DWH/Datalake de la empresa).
- `comprension_eda.ipynb`: análisis exploratorio y de comprensión de los datos
  (caracterización de variables, nulos, tipos, análisis univariable, bivariable y
  multivariable, y reglas de validación de datos).

## Dataset

`Base_de_datos.csv` — 10.763 registros de créditos históricos, 23 variables
(datos del crédito, del cliente y de comportamiento crediticio). Variable objetivo:
`Pago_atiempo` (1 = pagó a tiempo, 0 = no pagó a tiempo, ~5% de los casos).

## Avance 2 — Ingeniería de características y modelado

- `ft_engineering.py`: limpieza (reglas de validación sobre `tendencia_ingresos`
  y outliers de `salario_cliente`/`total_otros_prestamos`), atributos derivados
  (relación cuota/salario, relación deuda/salario, mes del préstamo, etc.) y un
  `ColumnTransformer` (numéricas → imputer+scaler, categóricas nominales →
  imputer+OneHotEncoder, `tendencia_ingresos` como ordinal → imputer+OrdinalEncoder).
- **Hallazgo importante:** la variable `puntaje` se excluyó del modelo por
  *data leakage* — separa casi perfectamente la variable objetivo, lo que indica
  que se calcula después de observar el comportamiento de pago del cliente y no
  estaría disponible al evaluar un solicitante nuevo.
- `model_training_evaluation.py`: entrena y compara Regresión Logística, Random
  Forest y XGBoost (todos ajustados para el fuerte desbalanceo de clases),
  usando `build_model()` y `summarize_classification()`. Selecciona el mejor
  modelo por ROC-AUC (Regresión Logística, ROC-AUC ≈ 0.68 tras remover el
  leakage) y lo guarda en `artifacts/best_model.pkl`.

## Avance 3 — Monitoreo y detección de data drift

**Caso de negocio:** detectar cambios en el perfil de los solicitantes de
crédito respecto a los datos con los que se entrenó el modelo, para anticipar
degradación de performance en producción antes de que impacte al negocio.

- `model_monitoring.py`: calcula métricas de drift por variable (KS test, PSI,
  Jensen-Shannon divergence para numéricas; Chi-cuadrado para categóricas),
  comparando una partición de referencia (créditos más antiguos) contra una
  partición "actual" (créditos más recientes), y evolución temporal por mes.
- **Hallazgo principal:** hay drift real en `plazo_meses`, `cuota_pactada`,
  `total_otros_prestamos`, `capital_prestado`, `salario_cliente`,
  `huella_consulta`, `saldo_total` y en las tres variables categóricas — es
  decir, el perfil de solicitantes cambió a lo largo del tiempo cubierto por el
  dataset. `edad_cliente`, `puntaje_datacredito` y `cant_creditosvigentes` se
  mantienen estables. Recomendación: monitorear de cerca la performance real del
  modelo y evaluar reentrenamiento periódico.
- Incluye una app en Streamlit (`streamlit run model_monitoring.py`) con tabla
  de métricas con semáforo, comparación de distribuciones, evolución temporal y
  recomendaciones automáticas.

## Avance 4 — Despliegue

- `model_deploy.py`: servicio FastAPI que carga `best_model.pkl` y
  `preprocessor.pkl`, expone `/health`, `/predict` (un solicitante) y
  `/predict/batch` (múltiples solicitantes), aplicando internamente el mismo
  pipeline de features que en el entrenamiento.
- `Dockerfile` / `.dockerignore`: imagen que empaqueta el código, las
  dependencias (`requirements.txt`) y los artefactos del modelo ya entrenado.
  Se corre con Uvicorn como servidor ASGI.

```bash
docker build -t credito-api .
docker run -p 8000:8000 credito-api
# POST http://localhost:8000/predict
```
## Registro de versiones

- V1.0.0: estructura base del repositorio.
- V1.0.1: Avance 1 — carga de datos y EDA.
- V1.1.0: Avance 2 — ingeniería de características y modelado.
- V1.1.1: Avance 3 — monitoreo y detección de data drift.
- V1.2.0: Avance 4 — despliegue con FastAPI y Docker.
