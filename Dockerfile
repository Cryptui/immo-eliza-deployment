# Starts from the python 3.10 official docker image
FROM python:3.10

# Install Java and clean up in one RUN command to keep image size down
# Java is a dependency for H2O, it is necessary to run H2O ML model 
RUN apt-get update && \
    apt-get install -y default-jre && \
    rm -rf /var/lib/apt/lists/*

# Define "/api" as the working directory
WORKDIR /api

# Copy the api directory into the container's working directory
COPY ./api /api

# Copy the models directory into the container
COPY ./models /models

# Copy the requirements.txt into the container's working directory
COPY requirements.txt /api

# Update pip and install dependencies from "requirements.txt"
# Use --no-cache-dir to reduce image size
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Set an environment variable for the model path
ENV MODEL_PATH=/models/GBM_4_AutoML

# Set an environment variable for the H2O server address
ENV H2O_SERVER=http://127.0.0.1:54321

# Expose both FastAPI and Streamlit ports
EXPOSE 8000
EXPOSE 8501

# Run the app using uvicorn for FastAPI and streamlit command for Streamlit
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
