# flask, scikit-learn, pandas, pickle-mixin
import pandas as pd
from flask import Flask, render_template, request
import pickle
import numpy as np
app = Flask(__name__)
data = pd.read_csv('Cleaned_data.csv')
pipe = pickle.load(open("RidgeModelNew.pkl", 'rb'))

# from sklearn.preprocessing import OneHotEncoder

# encoder = OneHotEncoder(handle_unknown='ignore')

@app.route('/')
def index():

    locations = sorted(data['location'].unique())
    return render_template('index.html', locations=locations)

@app.route('/predict', methods=['POST'])
def predict():
    location = request.form.get('location')
    bhk = request.form.get('bhk')
    bathrooms = request.form.get('bathrooms')
    sqft = request.form.get('total_sqft')

    print(location,bhk,bathrooms,sqft)
    input_data = pd.DataFrame([[location,sqft,bathrooms,bhk]], columns=['location', 'total_sqft', 'bathrooms', 'bhk'])

    prediction = pipe.predict(input_data)[0] * 1e5
    formatted_prediction = "{:,.2f}".format(prediction)
    return formatted_prediction


if __name__ == "__main__":
    app.run(debug=True, port=5001)

