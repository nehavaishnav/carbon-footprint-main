
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

# Load your dataset
data = pd.read_csv('CO2_Prediction_App/dataset.csv')  # Ensure the path to your CSV file is correct

# Define features and target variable
X = data.drop('CO2 Emissions(g/km)', axis=1)  # Feature set
y = data['CO2 Emissions(g/km)']  # Target variable

# Define categorical and numerical features
categorical_features = ['Make', 'Model', 'Vehicle Class', 'Transmission', 'Fuel Type']
numerical_features = ['Engine Size(L)', 'Cylinders', 
                     'Fuel Consumption City (L/100 km)', 
                     'Fuel Consumption Hwy (L/100 km)', 
                     'Fuel Consumption Comb (L/100 km)', 
                     'Fuel Consumption Comb (mpg)']

# Create a preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_features),  # Keep numerical features as is
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)  # One-hot encode categorical features
    ])

# Create a pipeline with preprocessing and model
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', RandomForestRegressor(random_state=42))  # Set random_state for reproducibility
])

# Split the dataset into training and testing sets before preprocessing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model using the pipeline
pipeline.fit(X_train, y_train)

# Evaluate the model
test_score = pipeline.score(X_test, y_test)
print(f'Model R^2 Score: {test_score:.2f}')

# Perform 5-fold cross-validation
cv_scores = cross_val_score(pipeline, X, y, cv=5)  # Adjust cv as needed
print(f'Cross-Validation Scores: {cv_scores}')
print(f'Mean Cross-Validation Score: {cv_scores.mean():.2f}')

# Make predictions on the test set
y_pred = pipeline.predict(X_test)

# Plotting Actual vs Predicted
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # Diagonal line
plt.title('Actual vs Predicted CO2 Emissions')
plt.xlabel('Actual CO2 Emissions (g/km)')
plt.ylabel('Predicted CO2 Emissions (g/km)')
plt.grid()
plt.show()

# Calculate errors
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)

print(f'Mean Absolute Error: {mae:.2f}')
print(f'Mean Squared Error: {mse:.2f}')
print(f'Root Mean Squared Error: {rmse:.2f}')

# Step 2: Test with Random Samples
# Select random samples from the dataset
random_samples = data.sample(n=5, random_state=42)  # Change n for more samples

# Prepare features for prediction
X_random = random_samples.drop('CO2 Emissions(g/km)', axis=1)

# Make predictions on the random samples
predictions = pipeline.predict(X_random)

# Display results
results = pd.DataFrame({
    'Actual CO2 Emissions': random_samples['CO2 Emissions(g/km)'],
    'Predicted CO2 Emissions': predictions
})

print("\nRandom Samples Predictions:")
print(results)

# Optional: Plot the predictions for random samples
plt.figure(figsize=(10, 6))
plt.barh(results.index, results['Actual CO2 Emissions'], color='blue', label='Actual')
plt.barh(results.index + 0.4, results['Predicted CO2 Emissions'], color='orange', label='Predicted')
plt.yticks(results.index, random_samples['Model'])  # Display vehicle models as y-ticks
plt.xlabel('CO2 Emissions (g/km)')
plt.title('Actual vs Predicted CO2 Emissions for Random Samples')
plt.legend()
plt.grid(axis='x')
plt.show()

import joblib

# Save the trained model to a file
joblib.dump(pipeline, 'model.pkl')
