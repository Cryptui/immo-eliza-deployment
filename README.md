# Immo Eliza - Model Deployment

Welcome to the `immo-eliza-deployment` repository where we have deployed a machine learning model that predicts real estate prices through an API endpoint. 
This solution consists of a FastAPI backend service deployed on Render and a frontend web application using Streamlit.

## Learning Objectives

- Understand the process of deploying a machine learning model through an API endpoint.
- Gain experience deploying an API to Render.
- Learn how to build a small web application using Streamlit.

## Project Overview

### Repository Structure

immo-eliza-deployment/
 
    ├── api/
    │   ├── init.py
    │   ├── app.py
    │   ├── predict.py
    │   └── requirements.txt
    ├── models/
    │   ├── GBM_4_AutoML
    ├── streamlit
    └── Dockerfile
    └── README.md

### FastAPI

The API is built with FastAPI and serves two endpoints:

- `/`: A GET request returns `"alive"` if the server is up.
- `/predict`: A POST request accepts property data in JSON format and returns a prediction in JSON format.

#### JSON Input Structure

    data: {
      "property_type": "string",
      "subproperty_type": "string",
      "region": "string",
      "province": "string",
      "locality": "string",
      "zip_code": "string",
      "construction_year": "string",
      "total_area_sqm": "string",
      "surface_land_sqm": "string",
      "nbr_frontages": "string",
      "nbr_bedrooms": "string",
      "equipped_kitchen": "string",
      "fl_furnished": "string",
      "fl_open_fire": "string",
      "fl_terrace": "string",
      "terrace_sqm": "string",
      "fl_garden": "string",
      "garden_sqm": "string",
      "fl_swimming_pool": "string",
      "fl_floodzone": "string",
      "state_building": "string",
      "primary_energy_consumption_sqm": "string",
      "epc": "string",
      "heating_type": "string",
      "fl_double_glazing": "string",
      "cadastral_income": "string"
    }

Output Structure eturn JSON a price in euros:

    {
      predicted_price: float 
    }

## Streamlit Web Application

The Streamlit app provides a user-friendly interface allowing non-technical stakeholders to input property details and receive price predictions.

## Deployment

The API is containerized with Docker and deployed on Render, while the Streamlit app is hosted on the Streamlit Community Cloud.

## Getting Started

### Prerequisites

- Python 3.6 or later
- Docker
- Render

### Setup Instructions

```bash
# Clone the repository:
git clone https://github.com/your_username/immo-eliza-deployment.git
cd immo-eliza-deployment

# Install dependencies:
pip install -r requirements.txt

# Run the FastAPI server:
uvicorn api.app:app --reload

# Launch the Streamlit app:
streamlit run streamlit/streamlit_immo_eliza.py




