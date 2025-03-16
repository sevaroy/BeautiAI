FROM python:3.9-slim

WORKDIR /app

# Install minimal system dependencies
RUN apt-get update && apt-get install -y \
    python3-dev \
    libatlas-base-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
COPY packages.txt .

# Install system packages from packages.txt
RUN apt-get update && xargs apt-get install -y < packages.txt

# Install dlib-bin instead of compiling dlib
RUN pip install dlib-bin

# Install other Python dependencies
RUN pip install -r requirements.txt

# Copy project files
COPY . .

# Expose port
EXPOSE 8501

# Start command
CMD ["streamlit", "run", "main.py"]
