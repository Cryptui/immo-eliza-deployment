# Use the official Python image.
FROM python:3.9-slim

# Set the working directory in the container.
WORKDIR /app

# Copy the requirements file into the container at /app.
COPY requirements.txt .

# Install any dependencies.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project directory into the container at /app.
COPY . /app


# Specify the command to run on container start.
# Make sure the path to the streamlit_immo_eliza.py is correct relative to the WORKDIR.
CMD ["streamlit", "run", "streamlit/streamlit_immo_eliza.py"]
