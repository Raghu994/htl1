from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Load individual decision tree models
tree_models = []
trees_directory = 'models/individual_trees'

# Load all individual tree models
for root, dirs, files in os.walk(trees_directory):
    for file in files:
        if file.endswith('.pkl'):
            tree_model = joblib.load(os.path.join(root, file))
            tree_models.append(tree_model)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the HTML form
        hotel = int(request.form['hotel'])
        lead_time = request.form['lead_time']
        arrival_date_week_number = request.form['arrival_date_week_number']
        adults = request.form['adults']
        children = request.form['children']
        babies = request.form['babies']
        days_in_waiting_list = request.form['days_in_waiting_list']
        adr = request.form['adr']
        required_car_parking_spaces = request.form['required_car_parking_spaces']
        total_of_special_requests = request.form['total_of_special_requests']
        meal = int(request.form['meal'])
        distribution_channel = int(request.form['distribution_channel'])
        reserved_room_type = int(request.form['reserved_room_type'])
        deposit_type = int(request.form['deposit_type'])

        input_data = np.array([[hotel, lead_time, arrival_date_week_number, adults, children, babies, days_in_waiting_list, adr, required_car_parking_spaces, total_of_special_requests, meal, distribution_channel, reserved_room_type, deposit_type]])

        # Initialize lists to store individual tree predictions and probabilities
        individual_tree_predictions = []
        individual_tree_probabilities = []

        # Make predictions using individual tree models and calculate probabilities
        for tree in tree_models:
            individual_tree_predictions.append(tree.predict(input_data))
            individual_tree_probabilities.append(tree.predict_proba(input_data)[:, 1])

        # Calculate the final probability as the average of individual tree probabilities
        final_probability = np.mean(individual_tree_probabilities)

        # Display both the final prediction (Cancelled/Not Cancelled) and accurate probability value
        final_prediction = 'Cancelled' if final_probability >= 0.5 else 'Not Cancelled'

        return render_template('index.html', prediction=final_prediction, probability=final_probability)
    except Exception as e:
        return jsonify({'error': str(e)}), 400  # Return a 400 Bad Request status code for errors

if __name__ == '__main__':
    app.run(debug=True)
