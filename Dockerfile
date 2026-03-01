FROM python:3.11-slim

WORKDIR /app

# 1. Instalare dependențe sistem necesare pentru Chromium
RUN apt-get update && apt-get install -y --no-install-recommends \
    # Dependențe Chromium esențiale
    libglib2.0-0 libnss3 libnspr4 libatk1.0-0 libatk-bridge2.0-0 \
    libcups2 libdrm2 libdbus-1-3 libexpat1 libxcb1 libxkbcommon0 \
    libx11-6 libxcomposite1 libxdamage1 libxext6 libxfixes3 libxrandr2 \
    libgbm1 libpango-1.0-0 libcairo2 libasound2 libatspi2.0-0 \
    # Utilitare
    curl unzip fonts-noto-color-emoji \
    # Curățare
    && rm -rf /var/lib/apt/lists/*

# 2. Instalare Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Instalare Playwright browsers (CRUCIAL!)
# Fără asta, chromium nu există în container
RUN python -m playwright install chromium
RUN python -m playwright install-deps chromium  # Dependențe suplimentare

# 4. Copiere cod
COPY main.py .

# 5. Creare director cache
RUN mkdir -p /app/cache

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]