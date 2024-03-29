import pandas as pd
import numpy as np
import h2o

# Initializes the H2O machine learning platform.
def initialize_h2o():
    h2o.init()

# Loads a pre-trained model from the specified path.
def load_model(model_path):
    return h2o.load_model(model_path)

# Fills missing values in the input DataFrame and replaces certain strings with standard codes.
def fill_missing_values(input_df, value=-1, replacements={'Missing': 'missing_info', 'MISSING': 'missing_info'}):
    # Replace NaN values with a default value (usually -1).
    df_filled = input_df.fillna(value)
    
    # Loop through the replacements dictionary and replace specified strings with a replacement value.
    for original, replacement in replacements.items():
        df_filled.replace(original, replacement, inplace=True)
    
    return df_filled

# Converts fields with Yes/No answers to boolean 1/0, and Unknown to NaN for model consumption.
def convert_boolean_fields(input_df):
    # List of fields that contain boolean data.
    boolean_fields = ['fl_furnished', 'fl_open_fire', 'fl_terrace', 'fl_garden', 'fl_swimming_pool', 'fl_floodzone', 'fl_double_glazing']
    
    # Convert Yes to 1, No to 0, and Unknown to NaN.
    for field in boolean_fields:
        input_df[field] = input_df[field].map({'Yes': 1, 'No': 0, 'Unknown': np.nan})
    
    return input_df

# Uses the provided model to predict the price based on the input DataFrame.
def predict_price(model, input_df):
    # Convert boolean fields and fill missing values in the input DataFrame.
    input_df = convert_boolean_fields(input_df)
    input_df = fill_missing_values(input_df)
    
    # Convert the DataFrame to an H2OFrame which is used by H2O for making predictions.
    preprocessed_hf = h2o.H2OFrame(input_df)
    
    # Make predictions using the model.
    predictions = model.predict(preprocessed_hf)
    
    # Extract the predicted value from the H2OFrame and return it.
    predicted_value = predictions.as_data_frame().values.flatten()[0]
    
    return predicted_value
