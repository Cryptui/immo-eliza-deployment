import sys
sys.path.append('D:/Github/Projects/immo-eliza-deployment/api')
import streamlit as st
import pandas as pd
import numpy as np
import h2o
from predict import load_model, initialize_h2o, fill_missing_values, predict_price

# Initialize the H2O server
initialize_h2o()

# Load the model
model = load_model('models/GBM_4_AutoML_2_20240321_133555')

# Streamlit app layout
st.title('Immo Eliza Real Estate Price Prediction')

st.write("""
### Please note:
This model provides the most accurate predictions for properties that have:
- Surface Land (sqm) up to 876.0
- Total Area (sqm) up to 270.0
- Number of Bedrooms up to 4
""")

# Define the form for user input
with st.form(key='input_form'):
    st.subheader('Property Details')

    # Create input fields
    property_type = st.selectbox('Property Type', ['', 'APARTMENT', 'HOUSE'])
    subproperty_type = st.selectbox('Subproperty Type', ['', 'DUPLEX', 'VILLA', 'EXCEPTIONAL_PROPERTY', 'FLAT_STUDIO', 'GROUND_FLOOR', 'PENTHOUSE', 'FARMHOUSE', 'APARTMENT_BLOCK', 'COUNTRY_COTTAGE', 'TOWN_HOUSE', 'SERVICE_FLAT', 'MANSION', 'MIXED_USE_BUILDING', 'MANOR_HOUSE', 'LOFT', 'BUNGALOW', 'KOT', 'CASTLE', 'CHALET', 'OTHER_PROPERTY', 'TRIPLEX'])
    region = st.selectbox('Region', ['', 'Flanders', 'Brussels-Capital', 'Wallonia'])
    province = st.text_input('Province')
    locality = st.text_input('Locality')
    zip_code = st.text_input('ZIP Code')
    construction_year = st.text_input('Construction Year')
    total_area_sqm = st.text_input('Total Area (sqm)')
    surface_land_sqm = st.text_input('Surface Land (sqm)')
    nbr_frontages = st.text_input('Number of Frontages')
    nbr_bedrooms = st.text_input('Number of Bedrooms')
    equipped_kitchen = st.selectbox('Equipped Kitchen', ['', 'INSTALLED', 'HYPER_EQUIPPED', 'NOT_INSTALLED', 'USA_UNINSTALLED', 'USA_HYPER_EQUIPPED', 'SEMI_EQUIPPED', 'USA_INSTALLED', 'USA_SEMI_EQUIPPED'])
    fl_furnished = st.radio('Furnished', ['Yes', 'No', 'Unknown'])
    fl_open_fire = st.radio('Open Fire', ['Yes', 'No', 'Unknown'])
    fl_terrace = st.radio('Terrace', ['Yes', 'No', 'Unknown'])
    terrace_sqm = st.text_input('Terrace Area (sqm)')
    fl_garden = st.radio('Garden', ['Yes', 'No', 'Unknown'])
    garden_sqm = st.text_input('Garden Area (sqm)')
    fl_swimming_pool = st.radio('Swimming Pool', ['Yes', 'No', 'Unknown'])
    fl_floodzone = st.radio('Flood Zone', ['Yes', 'No', 'Unknown'])
    state_building = st.selectbox('State of the Building', ['', 'AS_NEW', 'GOOD', 'TO_RENOVATE', 'TO_BE_DONE_UP', 'JUST_RENOVATED', 'TO_RESTORE'])
    primary_energy_consumption_sqm = st.text_input('Primary Energy Consumption (kWh/m²)')
    epc = st.selectbox('EPC Rating', ['', 'A++', 'A+', 'A', 'B', 'C', 'D', 'E', 'F', 'G'])
    heating_type = st.selectbox('Heating Type', ['', 'GAS', 'FUELOIL', 'PELLET', 'ELECTRIC', 'CARBON', 'SOLAR', 'WOOD'])
    fl_double_glazing = st.radio('Double Glazing', ['Yes', 'No', 'Unknown'])
    cadastral_income = st.text_input('Cadastral Income (€)')

    submit_button = st.form_submit_button(label='Predict Price')

if submit_button:
    input_data = {
        # your fields here
    }

    # Convert 'Yes'/'No' answers to 1/0 and handle 'Unknown'
    for field in ['fl_furnished', 'fl_open_fire', 'fl_terrace', 'fl_garden', 'fl_swimming_pool', 'fl_floodzone', 'fl_double_glazing']:
        input_data[field] = 1 if input_data[field] == 'Yes' else 0 if input_data[field] == 'No' else np.nan

    # Convert the dictionary to a pandas DataFrame
    input_df = pd.DataFrame([input_data])

    # Fill missing values and other preprocessing steps
    input_df = fill_missing_values(input_df)

    # Predict the price using the model
    preprocessed_hf = h2o.H2OFrame(input_df)
    predictions = model.predict(preprocessed_hf)
    predicted_value = predictions.as_data_frame().values.flatten()[0]

    # Display the prediction result
    st.success(f"The predicted real estate price is: €{predicted_value}")
