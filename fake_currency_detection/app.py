from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model
with open("fake_money_detector.pkl", "rb") as file:
    model = pickle.load(file)

# Define a route for the homepage
@app.route('/')
def home():
    return render_template("index.html")

# Define a route for predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get data from form
    data = request.form
    var = float(data['var'])
    skew = float(data['skew'])
    curt = float(data['curt'])
    entr = float(data['entr'])
    
    # Make prediction
    features = np.array([[var, skew, curt, entr]])
    prediction = model.predict(features)
    
    # Return result
    result = "Fake" if prediction[0] == 1 else "Real"
    return render_template("index.html", prediction_text=f"The currency is: {result}")

if __name__ == "__main__":
    app.run(debug=True)
