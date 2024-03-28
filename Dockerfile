# Use the official Python image.
FROM python:3.9-slim

# Set the working directory in the Docker container.
WORKDIR /app

# Copy the requirements file into the container at /app.
COPY requirements.txt /app/

# Install any dependencies.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project directory into the container at /app.
COPY . /app/

# Specify the command to run on container start.
CMD ["streamlit", "run", "streamlit/streamlit_immo_eliza.py"]
