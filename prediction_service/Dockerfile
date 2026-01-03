# Use lightweight Python image
FROM python:3.13-slim

# Set working directory
WORKDIR /app

# Install system build deps for some Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
	build-essential curl && \
	rm -rf /var/lib/apt/lists/*

# Install poetry
RUN pip install --no-cache-dir poetry

# Copy pyproject (use repository root as build context in the Makefile)
COPY pyproject.toml /app/

# Include metadata files that Poetry may validate (LICENSE, README)
COPY LICENSE /app/
COPY README.md /app/

# Configure poetry to install into the system environment and install dependencies
RUN poetry config virtualenvs.create false \
	&& poetry install --no-interaction --no-ansi --without dev --no-root

# Copy the prediction service application
COPY prediction_service /app

# Expose Flask port
EXPOSE 5000

# Run the Flask app using system Python (Poetry installed packages globally)
CMD ["python", "app.py"]
