# Use official Python 3.11 image
FROM python:3.11-slim

# Do not buffer Python output
ENV PYTHONUNBUFFERED=1

# Install system libs needed by Pillow and any image work
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libjpeg-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the FastAPI source code into the image
COPY . .

# Expose port (Cloud Run will override PORT environment variable)
EXPOSE 8080
ENV PORT=8080

# Start the FastAPI server via uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
