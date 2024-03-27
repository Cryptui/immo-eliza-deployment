from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import h2o
from typing import Optional

from predict import initialize_h2o, load_model, fill_missing_values, predict_price

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

def convert_boolean_values(input_data):
    for key, value in input_data.items():
        if value == 'Yes':
            input_data[key] = 1
        elif value == 'No':
            input_data[key] = 0
        elif value == 'Unknown':
            input_data[key] = np.nan
    return input_data

@app.post("/predict")
async def predict_endpoint(property_data: PropertyData):
    try:
        # Convert Pydantic object to dict
        input_dict = property_data.dict()
        
        # Convert 'Yes', 'No', 'Unknown' to 1, 0, np.nan
        input_dict = convert_boolean_values(input_dict)

        # Convert the dict to DataFrame
        input_df = pd.DataFrame([input_dict])

        # Preprocess the input data (fill missing values, etc.)
        input_df = fill_missing_values(input_df)

        # Convert DataFrame to H2OFrame and predict
        h2o_df = h2o.H2OFrame(input_df)
        prediction = model.predict(h2o_df)

        # Extract and return the predicted value
        predicted_value = prediction.as_data_frame().values.flatten()[0]
        return {"predicted_price": predicted_value}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
