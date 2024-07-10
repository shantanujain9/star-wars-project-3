import pandas as pd
import pickle

# Step 1: Load the Real Data
df_real = pd.read_csv('troop_movements10m.csv')

# Step 2: Data Cleaning
# Replace 'invalid_unit' with 'unknown'
df_real['unit_type'] = df_real['unit_type'].replace('invalid_unit', 'unknown')

# Fill missing location_x and location_y values using forward fill method
df_real['location_x'].fillna(method='ffill', inplace=True)
df_real['location_y'].fillna(method='ffill', inplace=True)

# Save the cleaned data to a Parquet file
df_real.to_parquet('troop_movements10m.parquet', engine='pyarrow')

# Step 3: Load the Model and Predict
# Load the trained model
with open('trained_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the cleaned data from the Parquet file
df_real_cleaned = pd.read_parquet('troop_movements10m.parquet')

# Convert the timestamp column to datetime
df_real_cleaned['timestamp'] = pd.to_datetime(df_real_cleaned['timestamp'])

# Extract useful components from the timestamp
df_real_cleaned['year'] = df_real_cleaned['timestamp'].dt.year
df_real_cleaned['month'] = df_real_cleaned['timestamp'].dt.month
df_real_cleaned['day'] = df_real_cleaned['timestamp'].dt.day
df_real_cleaned['hour'] = df_real_cleaned['timestamp'].dt.hour
df_real_cleaned['minute'] = df_real_cleaned['timestamp'].dt.minute
df_real_cleaned['second'] = df_real_cleaned['timestamp'].dt.second

# Drop the original timestamp column
df_real_cleaned.drop(columns=['timestamp'], inplace=True)

# Convert categorical features to numeric
df_real_encoded = pd.get_dummies(df_real_cleaned, columns=['home_world', 'unit_type'])

# Ensure the features match those used in the trained model
# You may need to align the columns with the training data
features_real = df_real_encoded.reindex(columns=features.columns, fill_value=0)

# Use the model to make predictions
df_real_encoded['is_resistance_pred'] = model.predict(features_real)

# Add the predicted values to the DataFrame
df_real_cleaned['is_resistance_pred'] = df_real_encoded['is_resistance_pred']

# Show the first few rows of the DataFrame with predictions
print(df_real_cleaned.head())
