import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from predict import load_model, get_input_data, predict_price, initialize_h2o

app = FastAPI()

# Initialize H2O
initialize_h2o()

# Get the absolute path to the directory containing the model
# Replace 'api/models/' with the actual relative path to the model directory
model_dir = os.path.join(os.path.dirname(__file__), 'models')

# Load the model using the absolute path
model_path = os.path.join(model_dir, 'GBM_4_AutoML_2_20240321_133555')
model = load_model(model_path)

class PropertyData(BaseModel):
    # Define the schema for the property data
    property_type: str
    subproperty_type: str
    region: str
    province: str
    locality: str
    zip_code: str
    construction_year: str
    total_area_sqm: str
    surface_land_sqm: str
    nbr_frontages: str
    nbr_bedrooms: str
    equipped_kitchen: str
    fl_furnished: str
    fl_open_fire: str
    fl_terrace: str
    terrace_sqm: str
    fl_garden: str
    garden_sqm: str
    fl_swimming_pool: str
    fl_floodzone: str
    state_building: str
    primary_energy_consumption_sqm: str
    epc: str
    heating_type: str
    fl_double_glazing: str
    cadastral_income: str

@app.get("/")
async def read_root():
    return {"message": "alive"}

@app.post("/predict")
async def predict(property_data: PropertyData):
    try:
        input_data = property_data.dict()
        input_df = get_input_data(input_data)
        predicted_price = predict_price(model, input_df)
        return {"predicted_price": predicted_price}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
