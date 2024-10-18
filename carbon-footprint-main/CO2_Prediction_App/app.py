from flask import Flask, render_template, request
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('model.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the form
    make = request.form['make']
    model_name = request.form['model']
    vehicle_class = request.form['vehicle_class']
    transmission = request.form['transmission']
    fuel_type = request.form['fuel_type']
    engine_size = float(request.form['engine_size'])
    cylinders = int(request.form['cylinders'])
    fuel_consumption_city = float(request.form['fuel_consumption_city'])
    fuel_consumption_hwy = float(request.form['fuel_consumption_hwy'])
    fuel_consumption_comb = float(request.form['fuel_consumption_comb'])

    # Create a DataFrame for the input data
    input_data = pd.DataFrame({
        'Make': [make],
        'Model': [model_name],
        'Vehicle Class': [vehicle_class],
        'Transmission': [transmission],
        'Fuel Type': [fuel_type],
        'Engine Size(L)': [engine_size],
        'Cylinders': [cylinders],
        'Fuel Consumption City (L/100 km)': [fuel_consumption_city],
        'Fuel Consumption Hwy (L/100 km)': [fuel_consumption_hwy],
        'Fuel Consumption Comb (L/100 km)': [fuel_consumption_comb],
        'Fuel Consumption Comb (mpg)': [fuel_consumption_comb * 0.425144]  # Convert L/100km to mpg
    })

    # Make predictions
    prediction = model.predict(input_data)[0]

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
