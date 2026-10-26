# Dockerfile  (Topic 2 onwards: package the model as a REST API service)

FROM python:3.13-slim

# Print logs immediately and do not write .pyc files
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install libraries first so this slow step is cached
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code and train the model during the build
COPY train.py app.py ./
RUN python train.py

# Do not run as root inside the container
RUN useradd --create-home appuser
USER appuser

# Document the port the API listens on
EXPOSE 5000

# Production web server (gunicorn) instead of Flask's development server
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "app:app"]
