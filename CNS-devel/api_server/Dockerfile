FROM python:3.10-slim

WORKDIR /app

# Copia solo requirements e installa dipendenze
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copia tutto il codice CNS-devel (incluso api_server e cns_external_images.py)
COPY .. /app

# Espone la porta 8000
EXPOSE 8000

CMD ["uvicorn", "api_server.main:app", "--host", "0.0.0.0", "--port", "8000"]
