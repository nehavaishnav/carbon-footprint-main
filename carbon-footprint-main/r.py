import pandas as pd

# Load your dataset
data = pd.read_csv('dataset.csv')  # Ensure the file path is correct

# Print columns to verify the DataFrame structure
print("Columns in the DataFrame:")
print(data.columns)

# Strip any extra spaces in the column names
data.columns = data.columns.str.strip()

# Check if the target column exists
if 'CO2 Emissions (g/km)' in data.columns:
    X = data.drop('CO2 Emissions (g/km)', axis=1)
    y = data['CO2 Emissions (g/km)']
else:
    print("The target column 'CO2 Emissions (g/km)' was not found.")
