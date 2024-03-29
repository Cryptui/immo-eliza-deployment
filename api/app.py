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
    property_type: Optional[str] = None
    subproperty_type: Optional[str] = None
    region: Optional[str] = None
    province: Optional[str] = None
    locality: Optional[str] = None
    zip_code: Optional[str] = None
    construction_year: Optional[str] = None
    total_area_sqm: Optional[str] = None
    surface_land_sqm: Optional[str] = None
    nbr_frontages: Optional[str] = None
    nbr_bedrooms: Optional[str] = None
    equipped_kitchen: Optional[str] = None
    fl_furnished: Optional[str] = None
    fl_open_fire: Optional[str] = None
    fl_terrace: Optional[str] = None
    terrace_sqm: Optional[str] = None
    fl_garden: Optional[str] = None
    garden_sqm: Optional[str] = None
    fl_swimming_pool: Optional[str] = None
    fl_floodzone: Optional[str] = None
    state_building: Optional[str] = None
    primary_energy_consumption_sqm: Optional[str] = None
    epc: Optional[str] = None
    heating_type: Optional[str] = None
    fl_double_glazing: Optional[str] = None
    cadastral_income: Optional[str] = None


@app.post("/predict")
async def predict_endpoint(property_data: PropertyData):
    try:
        # Convert Pydantic object to dict
        input_dict = property_data.dict()
        
        # Preprocess the input data
        input_df = pd.DataFrame([input_dict])
        # Preprocess the input DataFrame...
        
        # Predict using the model
        prediction = predict_price(model, input_df)

        # Return the predicted value
        return {"predicted_price": int(round(prediction,0))}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
