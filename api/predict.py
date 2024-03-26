import pandas as pd
import numpy as np
import h2o

def initialize_h2o():
    """
    Initialize the H2O server.
    """
    h2o.init()

def load_model(model_path):
    """
    Load a saved H2O model from a specified path.
    """
    return h2o.load_model(model_path)

def fill_missing_values(input_df, value=-1, replacements={'Missing': 'missing_info', 'MISSING': 'missing_info'}):
    """
    Fill missing values in a DataFrame.

    Parameters:
    - input_df: DataFrame to process.
    - value: Value to fill in for missing numeric data.
    - replacements: Dictionary mapping original missing value indicators to replacements.
    
    Returns:
    - DataFrame with missing values filled/replaced.
    """
    df_filled = input_df.fillna(value)
    for original, replacement in replacements.items():
        df_filled.replace(original, replacement, inplace=True)
    return df_filled

def get_input_data():
    """
    Prompt user to manually input the feature values for prediction, including handling for missing inputs and categorical options.
    
    Returns:
    - A single-row DataFrame with the input features.
    """
    # Example features with optional hints for categories, now including the options
    features_with_hints = {
        'property_type': 'Enter property_type (e.g., APARTMENT, HOUSE): ',
        'subproperty_type': 'Enter subproperty_type (Leave empty or  e.g., APARTMENT, HOUSE, DUPLEX, VILLA, EXCEPTIONAL_PROPERTY, FLAT_STUDIO, GROUND_FLOOR, PENTHOUSE, FARMHOUSE, APARTMENT_BLOCK, COUNTRY_COTTAGE, TOWN_HOUSE, SERVICE_FLAT, MANSION, MIXED_USE_BUILDING, MANOR_HOUSE, LOFT, BUNGALOW, KOT, CASTLE, CHALET, OTHER_PROPERTY, TRIPLEX): ',
        'region': 'Enter region (Leave empty or e.g., Flanders, Brussels-Capital, Wallonia): ',
        'province': 'Enter province (Leave empty or Antwerp, East Flanders, Brussels, Walloon Brabant, Flemish Brabant, LiÃ¨ge, West Flanders, Hainaut, Luxembourg, Limburg, Namur): ',
        'locality': 'Enter locality (Leave empty or e.g., Antwerp, Gent, Brussels, Turnhout, Nivelles, Halle-Vilvoorde, Liège, Brugge, Sint-Niklaas, Veurne, Verviers, Mechelen, Charleroi, Dendermonde, Bastogne, Leuven, Hasselt, Mons, Aalst, Tournai, Oostend, Oudenaarde, Philippeville, Kortrijk, Dinant, Ieper, Huy, Marche-en-Famenne, Namur, Maaseik, Mouscron, Diksmuide, Soignies, Neufchâteau, Arlon, Tongeren, Waremme, Thuin, Virton, Ath, Roeselare, Tielt, Eeklo): ',
        'zip_code': 'Enter zip_code (Leave empty or provide a value): ',
        'construction_year': 'Enter construction_year (Leave empty or provide a value): ',
        'total_area_sqm': 'Enter total_area_sqm (Leave empty or provide a value): ',
        'surface_land_sqm': 'Enter surface_land_sqm (Leave empty or provide a value): ',
        'nbr_frontages': 'Enter nbr_frontages (Leave empty or provide a value): ',
        'nbr_bedrooms': 'Enter nbr_bedrooms (Leave empty or provide a value): ',
        'equipped_kitchen': 'Enter equipped_kitchen (Leave empty or e.g., INSTALLED, HYPER_EQUIPPED, NOT_INSTALLED, USA_UNINSTALLED, USA_HYPER_EQUIPPED, SEMI_EQUIPPED, USA_INSTALLED, USA_SEMI_EQUIPPED): ',
        'fl_furnished': 'Enter fl_furnished (Leave empty or nan,  1.): ',
        'fl_open_fire': 'Enter fl_open_fire HOS(Leave empty or  nan,  1.): ',
        'fl_terrace': 'Enter fl_terrace (Leave empty or  nan,  1.): ',
        'terrace_sqm': 'Enter terrace_sqm (Leave empty or provide a value): ',
        'fl_garden': 'Enter fl_garden (Leave empty or  nan,  1.): ',
        'garden_sqm': 'Enter garden_sqm (Leave empty or provide a value): ',
        'fl_swimming_pool': 'Enter fl_swimming_pool (Leave empty or nan,  1.): ',
        'fl_floodzone': 'Enter fl_floodzone (Leave empty or  nan,  1.): ',
        'state_building': 'Enter state_building (Leave empty or e.g., AS_NEW, GOOD, TO_RENOVATE, TO_BE_DONE_UP, JUST_RENOVATED, TO_RESTORE): ',
        'primary_energy_consumption_sqm': 'Enter primary_energy_consumption_sqm (Leave empty or provide a value): ',
        'epc': 'Enter epc (Leave empty or  e.g., A++, A+, A, B, D, E, F, G): ',
        'heating_type': 'Enter heating_type (Leave empty or e.g., GAS, FUELOIL, PELLET, ELECTRIC, CARBON, SOLAR, WOOD): ',
        'fl_double_glazing': 'Enter fl_double_glazing (Leave empty or  nan,  1.): ',
        'cadastral_income': 'Enter cadastral_income (Leave empty or provide a value): '
    }

    # Skip these fields
    skip_fields = ['id', 'latitude', 'longitude']

    # Initialize a dictionary to hold user inputs, default to np.nan for missing
    input_data = {}

    # Iterate over features to get user inputs
    for feature, prompt in features_with_hints.items():
        if feature not in skip_fields:
            user_input = input(prompt).strip()
            if user_input:  # If the user provided a value, store it
                # Special handling for boolean fields
                if 'True' in prompt or 'False' in prompt:
                    input_data[feature] = user_input.lower() == 'true'
                else:
                    input_data[feature] = user_input
            else:
                input_data[feature] = np.nan
    
    # Convert the dictionary to a DataFrame
    input_df = pd.DataFrame([input_data])
    
    return input_df

def predict_price(model, input_df):
    """
    Predict the price for a single input instance.
    """
    # Preprocess the input data (fill missing values, etc.)
    preprocessed_data = fill_missing_values(input_df)
    
    # Convert the DataFrame to an H2OFrame
    preprocessed_hf = h2o.H2OFrame(preprocessed_data)
    
    # Make predictions
    predictions = model.predict(preprocessed_hf)
    
    # Extract and display the predicted value
    predicted_value = predictions.as_data_frame().values.flatten()[0]
    print(f"Predicted price: {predicted_value}")

if __name__ == "__main__":
    # Initialize H2O
    initialize_h2o()
    
    # Load the saved model
    model_path = 'models/GBM_4_AutoML_2_20240321_133555'
    model = load_model(model_path)
    
    # Get user input data
    input_df = get_input_data()
    
    # Predict the price and display the result
    predict_price(model, input_df)
