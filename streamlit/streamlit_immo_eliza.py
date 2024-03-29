import os
import streamlit as st
import pandas as pd
import numpy as np
import h2o
import sys

sys.path.append('/api')  # Adjust the path if predict.py is in a subdirectory

from pathlib import Path
from predict import load_model, initialize_h2o, predict_price, fill_missing_values

def add_bg_from_url():
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("https://miro.medium.com/v2/resize:fit:1400/0*cDRFtpTiOJFrfzS5.jpg");
            background-size: 100% 100%; /* Stretch to fill the screen width and height */
            background-attachment: fixed;
            background-repeat: no-repeat;
        }}
        
        .overlayBg {{
            position: fixed;
            top: 0;
            left: 0;
            height: 100vh;
            width: 100vw;
            background-color: rgba(240, 230, 90, 0.5); /* More opaque background */
            pointer-events: none;
            z-index: -1;
        }}

        /* Style adjustments for Streamlit widgets and markdown to have less transparent background */
        .stTextInput, .stSelectbox, .stDateInput, .stTimeInput, .stTextArea, .stMarkdown, .css-1b3qc1e {{
            background-color: rgba(255, 255, 255, 0.8); /* Solid background for inputs and text */
            border-radius: 5px;
            padding: 10px;
        }}

        /* Specific style for result area */
        .stAlert {{
            background-color: rgba(255, 255, 255, 0.8);
            border-radius: 5px;
            padding: 10px;
            margin-top: 5px;
        }}

        /* Style for buttons */
        .stButton > button {{
            border-radius: 5px;
            background-color: rgba(255, 255, 255, 0.8);
            padding: 10px;
            margin: 5px 0;
        }}

        /* Style for radio buttons and checkboxes */
        .stRadio > label, .stCheckbox > label {{
            background-color: rgba(255, 255, 255, 0.8);
            border-radius: 5px;
            padding: 5px 10px;
            display: block;
            margin-bottom: 5px;
        }}

        /* Style adjustments for Streamlit widgets container */
        .stWidget {{
            background-color: rgba(255, 255, 255, 0.8);
            border-radius: 5px;
            padding: 10px;
            margin-bottom: 10px;
        }}

        /* Ensure options like 'Yes', 'No', 'Unknown' are visible */
        .stRadio, .stCheckbox {{
            background-color: rgba(255, 255, 255, 0.8);
            padding: 10px;
            border-radius: 5px;
        }}

        /* Remove border around form fields */
        .stSelectbox, .stTextInput {{
            border: none !important;
        }}

        /* Adjust spacing between form fields */
        .stFormField {{
            margin-bottom: 10px !important;
        }}

        </style>
        <div class="overlayBg"></div>
        """,
        unsafe_allow_html=True
    )

add_bg_from_url()  # Call this function to add a background image and overlay


# Get the directory of the current file (streamlit_immo_eliza.py), then go up one level to the project root
project_root = Path(__file__).parent.parent

# Add the 'api' directory to sys.path
api_dir = project_root / 'api'
sys.path.append(str(api_dir))

# Initialize H2O server (only once)
@st.cache_resource
def initialize_h2o_server():
    h2o.init()
    return h2o


h2o_server = initialize_h2o_server()

# Use an environment variable for the model path
model_path = os.getenv("MODEL_PATH", "models/GBM_4_AutoML")
model = load_model(model_path)

# Streamlit app layout
st.title('Immo Eliza Real Estate Price Prediction')

st.write("""
**Instructions:**
1. Default values are pre-filled for test purposes. Adjust them as needed.
2. Fill in as many fields as possible for a more accurate price prediction.
3. Leave fields blank if you don't have any information or if it's not applicable.

**Tips for accurate predictions:**
- Provide data for properties with the following characteristics:
    - Surface Land (sqm) up to 876.0
    - Total Area (sqm) up to 270.0
    - Number of Bedrooms up to 4.

**Note:** 
The first price prediction may take a bit longer after pressing the button as the model processes the input data.

""")


# Define the form for user input
with st.form(key='input_form'):
    st.subheader('Property Details')

    # Create input fields
    property_type = st.selectbox('Property Type', ['APARTMENT', 'HOUSE', ''])
    subproperty_type = st.selectbox('Subproperty Type', ['APARTMENT', 'HOUSE', '', 'VILLA', 'GROUND_FLOOR', 'APARTMENT_BLOCK', 'BUNGALOW', 'CASTLE', 'CHALET', 'COUNTRY_COTTAGE', 'DUPLEX', 'EXCEPTIONAL_PROPERTY', 'FARMHOUSE', 'FLAT_STUDIO', 'GROUND_FLOOR', 'KOT', 'LOFT', 'MANOR_HOUSE', 'MANSION', 'MIXED_USE_BUILDING', 'OTHER_PROPERTY', 'PENTHOUSE', 'SERVICE_FLAT', 'TOWN_HOUSE', 'TRIPLEX', 'VILLA'])
    region = st.selectbox('Region', ['Flanders', 'Brussels-Capital', '', 'Wallonia'])
    province_options = ['Antwerp', 'Brussels', '', 'East Flanders', 'Flemish Brabant', 'Hainaut', 'LiÃƒÂ¨ge', 'Limburg', 'Luxembourg', 'Namur', 'Walloon Brabant', 'West Flanders']  
    province = st.selectbox('Province', province_options)
    locality_options = ['Antwerp', 'Aalst', '', 'Arlon', 'Ath', 'Bastogne', 'Brugge', 'Brussels', 'Charleroi', 'Dendermonde', 'Diksmuide', 'Dinant', 'Eeklo', 'Gent', 'Halle-Vilvoorde', 'Hasselt', 'Huy', 'Ieper', 'Kortrijk', 'Leuven', 'LiÃƒÂ¨ge', 'Maaseik', 'Marche-en-Famenne', 'Mechelen', 'Mons', 'Mouscron', 'Namur', 'NeufchÃƒÂ¢teau', 'Nivelles', 'Oostend', 'Oudenaarde', 'Philippeville', 'Roeselare', 'Sint-Niklaas', 'Soignies', 'Thuin', 'Tielt', 'Tongeren', 'Tournai', 'Turnhout', 'Verviers', 'Veurne', 'Virton', 'Waremme']
    locality = st.selectbox('Locality', locality_options)
    zip_code = st.text_input('ZIP Code (between 1000-9992)', '1000')
    if zip_code:
        try:
            zip_code = int(zip_code)
            if not (1000 <= zip_code <= 9992):
                st.warning('ZIP code must be between 1000 and 9992')
        except ValueError:
            st.warning('Invalid ZIP code. Please enter a valid number.')
    construction_year = st.text_input('Construction Year', '1995')
    total_area_sqm = st.text_input('Total Area (sqm)', '120')
    surface_land_sqm = st.text_input('Surface Land (sqm)')
    nbr_frontages = st.text_input('Number of Frontages', '2')
    nbr_bedrooms = st.text_input('Number of Bedrooms', '2')
    equipped_kitchen = st.selectbox('Equipped Kitchen', ['INSTALLED', 'HYPER_EQUIPPED', '', 'missing_info', 'NOT_INSTALLED', 'SEMI_EQUIPPED', 'USA_HYPER_EQUIPPED', 'USA_INSTALLED', 'USA_SEMI_EQUIPPED', 'USA_UNINSTALLED'])
    fl_furnished = st.radio('Furnished', ['Yes', 'No', 'Unknown'], index=1)
    fl_open_fire = st.radio('Open Fire', ['Yes', 'No', 'Unknown'], index=1)
    fl_terrace = st.radio('Terrace', ['Yes', 'No', 'Unknown'], index=0)
    terrace_sqm = st.text_input('Terrace Area (sqm)', '10')
    fl_garden = st.radio('Garden', ['Yes', 'No', 'Unknown'], index=1)
    garden_sqm = st.text_input('Garden Area (sqm)')
    fl_swimming_pool = st.radio('Swimming Pool', ['Yes', 'No', 'Unknown'], index=1)
    fl_floodzone = st.radio('Flood Zone', ['Yes', 'No', 'Unknown'], index=1)
    state_building = st.selectbox('State of the Building', ['GOOD', 'AS_NEW', '', 'TO_RENOVATE', 'TO_BE_DONE_UP', 'JUST_RENOVATED', 'TO_RESTORE'])
    primary_energy_consumption_sqm = st.text_input('Primary Energy Consumption (kWh/m²)', '400')
    epc = st.selectbox('EPC Rating', ['B', 'A++', '', 'A+', 'A', 'C', 'D', 'E', 'F', 'G'])
    heating_type = st.selectbox('Heating Type', ['GAS', 'FUELOIL', '', 'PELLET', 'ELECTRIC', 'CARBON', 'SOLAR', 'WOOD'])
    fl_double_glazing = st.radio('Double Glazing', ['Yes', 'No', 'Unknown'], index=0)
    cadastral_income = st.text_input('Cadastral Income (€)', '1200')

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
    numerical_fields = ['total_area_sqm', 'zip_code', 'construction_year', 'surface_land_sqm', 'nbr_frontages', 'nbr_bedrooms', 
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


