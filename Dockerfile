# Use the official Python image.
FROM python:3.9

# Set the working directory in the Docker container.
WORKDIR /app

# Copy the requirements file into the container at /app.
COPY requirements.txt .

# Install any dependencies.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project directory into the container at /app.
COPY . .

# Specify the command to run on container start.
CMD ["streamlit", "run", "streamlit/streamlit_immo_eliza.py"]
