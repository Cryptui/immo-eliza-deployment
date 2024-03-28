# Starts from the python 3.10 official docker image
FROM python:3.10

# Install Java
RUN apt-get update && \
    apt-get install -y default-jre

# Create a folder "/api" at the root of the image
RUN mkdir /api

# Define "/api" as the working directory
WORKDIR /api

# Copy all the files from the current directory to "/api"
COPY . /api

# Update pip
RUN pip install --upgrade pip

# Install dependencies from "requirements.txt"
RUN pip install -r requirements.txt

# Set an environment variable for the model path
ENV MODEL_PATH=/models/GBM_4_AutoML

# Run the app
# Set host to 0.0.0.0 to make it run on the container's network
CMD ["uvicorn", "app:app", "--host", "0.0.0.0"]
