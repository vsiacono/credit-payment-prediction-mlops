# mlops_pipeline — Predicción de comportamiento de pago de créditos

Proyecto Integrador Módulo 5 (Data Science). Modelo de aprendizaje automático que
predice si un nuevo solicitante de crédito pagará a tiempo (`Pago_atiempo`), usando
información histórica de créditos de una empresa financiera.

## Estructura del repositorio

```
mlops_pipeline/
├── src/
│   ├── config.json                     # configuración del proyecto (project_code, etc.)
│   ├── Cargar_datos.ipynb              # carga del dataset de ejemplo (.csv)
│   ├── comprension_eda.ipynb           # análisis exploratorio de datos
│   ├── ft_engineering.py               # ingeniería de features (próximo avance)
│   ├── model_training_evaluation.py    # entrenamiento y evaluación (próximo avance)
│   ├── model_deploy.py                 # despliegue del modelo (próximo avance)
│   └── model_monitoring.py             # monitoreo en producción (próximo avance)
├── Base_de_datos.csv
├── requirements.txt
├── .gitignore
└── readme.md
```

Esta estructura es fija y no debe modificarse: los procesos de despliegue a
producción están automatizados vía pipelines de Jenkins que dependen de ella.

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
