from flask import Flask, request, render_template, send_from_directory
import pickle
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

with open('rf_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('feature_columns.pkl', 'rb') as f:
    expected_columns = pickle.load(f)

numeric_cols = expected_columns
model_metrics = {
    'Linear Regression': {'MAE': 0.53, 'MSE': 0.56, 'R2': 0.58},
    'Random Forest': {'MAE': 0.32, 'MSE': 0.24, 'R2': 0.81},
    'XGBoost': {'MAE': 0.31, 'MSE': 0.23, 'R2': 0.82},
    'Tuned Random Forest': {'MAE': 0.31, 'MSE': 0.24, 'R2': 0.81}
}
plots = {
    'pair_plot_numeric': 'pair_plot_numeric.png',
    'house_value_distribution': 'house_value_distribution.png',
    'log_house_value_distribution': 'log_house_value_distribution.png',
    'house_value_by_age_violin': 'house_value_by_age_violin.png',
    'residuals_linear_regression': 'residuals_linear_regression.png',
    'residuals_random_forest': 'residuals_random_forest.png',
    'residuals_xgboost': 'residuals_xgboost.png',
    'residuals_tuned_random_forest': 'residuals_tuned_random_forest.png',
    'model_comparison_mae': 'model_comparison_mae.png',
    'model_comparison_r2': 'model_comparison_r2.png',
    'model_metrics_comparison': 'model_metrics_comparison.png'
}
@app.route('/files/<filename>')
def serve_file(filename):
    return send_from_directory(os.getcwd(), filename)

@app.route('/')
def home():
    return render_template('index.html', prediction=None, plots=plots, metrics=model_metrics)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        med_inc = float(request.form['med_inc'])
        house_age = float(request.form['house_age'])
        ave_rooms = float(request.form['ave_rooms'])
        ave_bedrms = float(request.form['ave_bedrms'])
        population = float(request.form['population'])
        ave_occup = float(request.form['ave_occup'])
        latitude = float(request.form['latitude'])
        longitude = float(request.form['longitude'])
    except ValueError:
        return render_template('index.html', prediction="Error: Please enter valid numeric values for all fields.", 
                             plots=plots, metrics=model_metrics)
    input_data = pd.DataFrame({
        'MedInc': [med_inc],
        'HouseAge': [house_age],
        'AveRooms': [ave_rooms],
        'AveBedrms': [ave_bedrms],
        'Population': [population],
        'AveOccup': [ave_occup],
        'Latitude': [latitude],
        'Longitude': [longitude]
    }
    for col in expected_columns:
        if col not in input_data.columns:
            input_data[col] = 0
    input_data = input_data[expected_columns]
    input_data[numeric_cols] = scaler.transform(input_data[numeric_cols])
    try:
        prediction = model.predict(input_data)[0]
        return render_template('index.html', prediction=f'Predicted Median House Value: ${prediction * 100_000:.2f}', 
                             plots=plots, metrics=model_metrics)
    except Exception as e:
        return render_template('index.html', prediction=f'Error during prediction: {str(e)}', 
                             plots=plots, metrics=model_metrics)

if __name__ == '__main__':
    app.run(debug=True)