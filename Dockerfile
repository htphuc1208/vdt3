# TelcoMAS — containerised demo (Streamlit dashboard by default).
FROM python:3.12-slim

WORKDIR /app

# install deps first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# app code
COPY . .

EXPOSE 8501

# Provide LLM credentials at run time, e.g.:
#   docker run --rm -p 8501:8501 --env-file .env telco-mas
ENV PYTHONUNBUFFERED=1
CMD ["streamlit", "run", "apps/dashboard.py", "--server.address=0.0.0.0", "--server.port=8501"]
