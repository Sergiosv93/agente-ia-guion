FROM python:3.11-slim

WORKDIR /code

# Torch con CUDA 12.8 (misma build que en el venv local)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu128

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./app ./app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
