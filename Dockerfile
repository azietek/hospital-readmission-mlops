# Official Python image as a base (slim version takes up less space)
FROM python:3.12-slim

# Set the working directory inside the container to /app
WORKDIR /app

# Installing build-essential package to compile C/C++ code, which is required for some Python packages like XGBoost.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copying requirements.txt to the working directory in the container
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copying the source code of the application to the container
COPY src/ /app/src/

# Informing on which port the application will run inside the container
EXPOSE 8000

# Starting the FastAPI application, changing the host to 0.0.0.0, so that the API is accessible outside the container
CMD ["uvicorn", "src.predict_api:app", "--host", "0.0.0.0", "--port", "8000"]