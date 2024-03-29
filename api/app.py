from dotenv import load_dotenv
load_dotenv()  # This loads the variables from .env into the environment.

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import h2o
from typing import Optional
import os

from predict import load_model as predict_load_model, fill_missing_values, predict_price

# Connect to the H2O server using the environment variable
h2o_server = os.environ.get('H2O_SERVER', 'http://localhost:54321')
h2o.init(url=h2o_server)

# Use an environment variable for the model path, with a default if not set
model_path = os.getenv('MODEL_PATH', '/models/GBM_4_AutoML')

# Load the model using the adjusted function name to avoid naming conflict
model = predict_load_model(model_path)

app = FastAPI()

@app.get("/")
async def read_root():
    return {"message": "Welcome to your FastAPI application!"}

class PropertyData(BaseModel):
    property_type: Optional[str] = "HOUSE"
    subproperty_type: Optional[str] = "VILLA"
    region: Optional[str] = "Flanders"
    province: Optional[str] = "Antwerp"
    locality: Optional[str] = "Antwerp"
    zip_code: Optional[str] = "2000"
    construction_year: Optional[str] = "1990"
    total_area_sqm: Optional[str] = "200"
    surface_land_sqm: Optional[str] = None
    nbr_frontages: Optional[str] = "2"
    nbr_bedrooms: Optional[str] = "3"
    equipped_kitchen: Optional[str] = "INSTALLED"
    fl_furnished: Optional[str] = "No"
    fl_open_fire: Optional[str] = "Yes"
    fl_terrace: Optional[str] = "Yes"
    terrace_sqm: Optional[str] = None
    fl_garden: Optional[str] = "Yes"
    garden_sqm: Optional[str] = None
    fl_swimming_pool: Optional[str] = "No"
    fl_floodzone: Optional[str] = "No"
    state_building: Optional[str] = "GOOD"
    primary_energy_consumption_sqm: Optional[str] = "200"
    epc: Optional[str] = "B"
    heating_type: Optional[str] = "GAS"
    fl_double_glazing: Optional[str] = "Yes"
    cadastral_income: Optional[str] = "1500"


@app.post("/predict")
async def predict_endpoint(property_data: PropertyData):
    # Convert Pydantic object to dict
    input_dict = property_data.dict()
    try:
        # Preprocess the input data
        input_df = pd.DataFrame([input_dict])
        # Preprocess the input DataFrame...
        
        # Predict using the model
        prediction = predict_price(model, input_df)

        # Return the predicted value
        return {"predicted_price": int(round(prediction, 0))}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

