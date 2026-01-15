# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Create a non-root user with UID 1000 (Required by Hugging Face)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:${PATH}"

# Install Python dependencies
COPY --chown=user:user requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY --chown=user:user . .

# Hugging Face expects port 7860
EXPOSE 7860

# Run the application on port 7860 specifically
# Using 1 worker to maximize memory availability for heavy processing
# Increased timeout to 600 seconds (10 mins) for long-running analysis
CMD ["gunicorn", "main:app", "-w", "1", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:7860", "--timeout", "600"]
