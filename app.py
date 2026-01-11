import joblib
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('logistic_regression_model.joblib')   
scaler = joblib.load('scaler.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            data = request.get_json(force=True) # Get data posted as JSON
            # Convert input data to a DataFrame
            input_df = pd.DataFrame([data])
            
            # Ensure the order of columns matches the training data
            # Assuming the columns of the original X (features) are known
            # X.columns are 'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
            expected_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
            
            # Reindex to ensure correct column order and handle missing columns with NaN
            input_df = input_df.reindex(columns=expected_columns, fill_value=0)

            # Scale the input features
            scaled_data = scaler.transform(input_df)
            
            # Make prediction
            prediction = model.predict(scaled_data)
            prediction_proba = model.predict_proba(scaled_data)

            # Return prediction as JSON
            return jsonify({
                'prediction': int(prediction[0]),
                'prediction_probability_class_0': prediction_proba[0][0],
                'prediction_probability_class_1': prediction_proba[0][1]
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # For development, run with debug=True. In production, use a production-ready server.
    app.run(host='0.0.0.0', port=5000)
