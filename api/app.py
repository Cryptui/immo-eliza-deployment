from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import h2o
from typing import Optional

from .predict import initialize_h2o, load_model, fill_missing_values, convert_boolean_fields, predict_price


app = FastAPI()

# Initialize H2O
initialize_h2o()

# Assuming your model is in the relative directory specified
model_path = 'D:/Github/Projects/immo-eliza-deployment/models/GBM_4_AutoML_2_20240321_133555'
model = load_model(model_path)

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
        input_df = fill_missing_values(input_df)
        input_df = convert_boolean_fields(input_df)

        # Predict using the model
        prediction = predict_price(model, input_df)

        # Return the predicted value
        return {"predicted_price": int(round(prediction,0))}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
