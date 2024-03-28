import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import h2o

# Adjust the path to include your 'api' directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'api')))

from api.predict import load_model, initialize_h2o, predict_price, fill_missing_values

# Initialize H2O server (only once)
@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def initialize_h2o_server():
    h2o.init()
    return h2o

h2o_server = initialize_h2o_server()

# Load model (only once)
@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def load_h2o_model(model_path):
    return h2o.load_model(model_path)

#model = load_h2o_model('D:/Github/Projects/immo-eliza-deployment/models/GBM_4_AutoML_2_20240321_133555')
model = load_h2o_model('/app/models/GBM_4_AutoML_2_20240321_133555')

# Streamlit app layout
st.title('Immo Eliza Real Estate Price Prediction')

st.write("""
### Please note:
This model provides the most accurate predictions for properties that have:
- Surface Land (sqm) up to 876.0
- Total Area (sqm) up to 270.0
- Number of Bedrooms up to 4
- Leave values blank if you don't have any information or is not applicable
""")


# Define the form for user input
with st.form(key='input_form'):
    st.subheader('Property Details')

    # Create input fields
    property_type = st.selectbox('Property Type', ['', 'APARTMENT', 'HOUSE'])
    subproperty_type = st.selectbox('Subproperty Type', ['', 'APARTMENT', 'HOUSE', 'VILLA', 'GROUND_FLOOR', 'APARTMENT_BLOCK', 'BUNGALOW', 'CASTLE', 'CHALET', 'COUNTRY_COTTAGE', 'DUPLEX', 'EXCEPTIONAL_PROPERTY', 'FARMHOUSE', 'FLAT_STUDIO', 'GROUND_FLOOR', 'KOT', 'LOFT', 'MANOR_HOUSE', 'MANSION', 'MIXED_USE_BUILDING', 'OTHER_PROPERTY', 'PENTHOUSE', 'SERVICE_FLAT', 'TOWN_HOUSE', 'TRIPLEX', 'VILLA'])
    region = st.selectbox('Region', ['', 'Flanders', 'Brussels-Capital', 'Wallonia'])
    province_options = ['', 'Antwerp', 'Brussels', 'East Flanders', 'Flemish Brabant', 'Hainaut', 'LiÃƒÂ¨ge', 'Limburg', 'Luxembourg', 'Namur', 'Walloon Brabant', 'West Flanders']  
    province = st.selectbox('Province', province_options)
    locality_options = ['', 'Aalst', 'Antwerp', 'Arlon', 'Ath', 'Bastogne', 'Brugge', 'Brussels', 'Charleroi', 'Dendermonde', 'Diksmuide', 'Dinant', 'Eeklo', 'Gent', 'Halle-Vilvoorde', 'Hasselt', 'Huy', 'Ieper', 'Kortrijk', 'Leuven', 'LiÃƒÂ¨ge', 'Maaseik', 'Marche-en-Famenne', 'Mechelen', 'Mons', 'Mouscron', 'Namur', 'NeufchÃƒÂ¢teau', 'Nivelles', 'Oostend', 'Oudenaarde', 'Philippeville', 'Roeselare', 'Sint-Niklaas', 'Soignies', 'Thuin', 'Tielt', 'Tongeren', 'Tournai', 'Turnhout', 'Verviers', 'Veurne', 'Virton', 'Waremme']
    locality = st.selectbox('Locality', locality_options)
    zip_code = st.text_input('ZIP Code (between 1000-9992)')
    construction_year = st.text_input('Construction Year')
    total_area_sqm = st.text_input('Total Area (sqm)')
    surface_land_sqm = st.text_input('Surface Land (sqm)')
    nbr_frontages = st.text_input('Number of Frontages')
    nbr_bedrooms = st.text_input('Number of Bedrooms')
    equipped_kitchen = st.selectbox('Equipped Kitchen', ['', 'HYPER_EQUIPPED', 'INSTALLED', 'missing_info', 'NOT_INSTALLED', 'SEMI_EQUIPPED', 'USA_HYPER_EQUIPPED', 'USA_INSTALLED', 'USA_SEMI_EQUIPPED', 'USA_UNINSTALLED'])
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
        'property_type': property_type,
        'subproperty_type': subproperty_type,
        'region': region,
        'province': province,
        'locality': locality,
        'zip_code': zip_code,
        'construction_year': construction_year,
        'total_area_sqm': total_area_sqm,
        'surface_land_sqm': surface_land_sqm,
        'nbr_frontages': nbr_frontages,
        'nbr_bedrooms': nbr_bedrooms,
        'equipped_kitchen': equipped_kitchen,
        'fl_furnished': 'Yes' if fl_furnished == 'Yes' else 'No' if fl_furnished == 'No' else np.nan,
        'fl_open_fire': 'Yes' if fl_open_fire == 'Yes' else 'No' if fl_open_fire == 'No' else np.nan,
        'fl_terrace': 'Yes' if fl_terrace == 'Yes' else 'No' if fl_terrace == 'No' else np.nan,
        'terrace_sqm': terrace_sqm,
        'fl_garden': 'Yes' if fl_garden == 'Yes' else 'No' if fl_garden == 'No' else np.nan,
        'garden_sqm': garden_sqm,
        'fl_swimming_pool': 'Yes' if fl_swimming_pool == 'Yes' else 'No' if fl_swimming_pool == 'No' else np.nan,
        'fl_floodzone': 'Yes' if fl_floodzone == 'Yes' else 'No' if fl_floodzone == 'No' else np.nan,
        'state_building': state_building,
        'primary_energy_consumption_sqm': primary_energy_consumption_sqm,
        'epc': epc,
        'heating_type': heating_type,
        'fl_double_glazing': 'Yes' if fl_double_glazing == 'Yes' else 'No' if fl_double_glazing == 'No' else np.nan,
        'cadastral_income': cadastral_income
    }

    # After collecting all input data into input_data dictionary but before converting it to DataFrame
    numerical_fields = ['total_area_sqm', 'surface_land_sqm', 'nbr_frontages', 'nbr_bedrooms', 
                        'terrace_sqm', 'garden_sqm', 'primary_energy_consumption_sqm', 'cadastral_income']

    # Replace empty strings with a default value, for example -1
    for field in numerical_fields:
        input_data[field] = float(input_data[field]) if input_data[field] else -1

    # Now, input_data should have -1 or the appropriate value for each numerical field if they were left empty


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
    st.success(f"The predicted real estate price is: €{int(round(predicted_value,0))}")
