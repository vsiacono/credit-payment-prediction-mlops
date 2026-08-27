# Imagen para el despliegue del modelo de predicción de comportamiento de pago
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Código fuente y artefactos del modelo ya entrenado
COPY src/ ./src/
COPY artifacts/ ./artifacts/
COPY Base_de_datos.csv .

WORKDIR /app/src

EXPOSE 8000

CMD ["uvicorn", "model_deploy:app", "--host", "0.0.0.0", "--port", "8000"]
