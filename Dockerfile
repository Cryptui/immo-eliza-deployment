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

# Copy the streamlit directory into the container
COPY ./streamlit /api/streamlit

# Update pip and install dependencies from "requirements.txt"
# Use --no-cache-dir to reduce image size
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Install Streamlit
RUN pip install streamlit

# Set an environment variable for the model path
ENV MODEL_PATH=/models/GBM_4_AutoML

# Set an environment variable for the H2O server address
ENV H2O_SERVER=http://127.0.0.1:54321

# Add Streamlit installation directory to PATH
ENV PATH="/usr/local/bin:${PATH}"

# Expose both FastAPI and Streamlit ports
EXPOSE 8000
EXPOSE 8501

# Run the Streamlit app at the root URL
CMD ["streamlit", "run", "/api/streamlit/streamlit_immo_eliza.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]
