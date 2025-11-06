# Ruaidhrí
# Creating the flask app to be hosted which will communicate with the keras model

import tensorflow.lite as tflite
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np

FEATURES = ["Age", "Gender", "Education Level", "BMI", "Physical Activity Level", "Smoking Status",
            "Alcohol Consumption", "Diabetes", "Hypertension", "Cholesterol Level", "Family History of Alzheimer’s",
            "Cognitive Test Score", "Depression Level", "Sleep Quality", "Dietary Habits", "Air Pollution Exposure",
            "Employment Status", "Marital Status", "Genetic Risk Factor (APOE-ε4 allele)", "Social Engagement Level",
            "Income Level", "Stress Levels", "Urban vs Rural Living"]

app = Flask(__name__)
interpreter = tflite.Interpreter("alzheimers_model.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


@app.route("/")
def status():
    return "<h1>Flask application is running</h1>"


@app.route("/submit", methods=["POST"])
def submit_data():
    try:
        user_data = request.get_json()  # Retrieves the request that was made
        data_values = user_data["data"]  # Will get the actual numerical values that were sent
        data_dictionary = dict(zip(FEATURES, data_values))  # Converts it to dictionary, so it can be made a DataFrame
        testing_data = pd.DataFrame([data_dictionary])  # Converts it to a DataFrame so that the model can use it
        input_data = np.array(testing_data, dtype=np.float32)

        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]

        return jsonify({"alzheimers_risk": float(prediction)}), 200  # 200 indicates successful request
    except Exception as e:
        return jsonify({"error": str(e)}), 400  # 400 indicates a bad request


if __name__ == "__main__":
    app.run()

"""
To make a request use the following:
    curl -X POST -H "Content-Type: application/json" -d "{\"data\": [VALUES]}" http://127.0.0.1:5000/submit

Make sure to replace "VALUES" with an array like these:
    [20, 1, 19, 20.0, 2, 0, 0, 0, 0, 0, 0, 99, 0, 2, 2, 0, 1, 1, 0, 2, 2, 0, 0]
    [90, 0, 0, 31.0, 0, 2, 2, 1, 1, 1, 1, 30, 2, 0, 0, 2, 2, 2, 1, 0, 0, 2, 1]
"""
