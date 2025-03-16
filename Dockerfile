FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
COPY packages.txt .
RUN apt-get update && xargs apt-get install -y < packages.txt

# Install Python dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy project files
COPY . .

# Set environment variables
ENV FLASK_ENV=production

# Expose port
EXPOSE 8501

# Start command
CMD ["streamlit", "run", "main.py"]
