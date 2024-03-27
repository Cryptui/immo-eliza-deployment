import pandas as pd
import numpy as np
import h2o

def initialize_h2o():
    h2o.init()

def load_model(model_path):
    return h2o.load_model(model_path)

def fill_missing_values(input_df, value=-1, replacements={'Missing': 'missing_info', 'MISSING': 'missing_info'}):
    df_filled = input_df.fillna(value)
    for original, replacement in replacements.items():
        df_filled.replace(original, replacement, inplace=True)
    return df_filled

def convert_boolean_fields(input_df):
    boolean_fields = ['fl_furnished', 'fl_open_fire', 'fl_terrace', 'fl_garden', 'fl_swimming_pool', 'fl_floodzone', 'fl_double_glazing']
    for field in boolean_fields:
        input_df[field] = input_df[field].map({'Yes': 1, 'No': 0, 'Unknown': np.nan})
    return input_df

def predict_price(model, input_df):
    input_df = convert_boolean_fields(input_df)
    input_df = fill_missing_values(input_df)
    preprocessed_hf = h2o.H2OFrame(input_df)
    predictions = model.predict(preprocessed_hf)
    predicted_value = predictions.as_data_frame().values.flatten()[0]
    return predicted_value

