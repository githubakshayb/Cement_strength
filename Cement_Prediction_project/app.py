from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import pickle

model = pickle.load(open('model.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    MaterialQuantity = float(request.form['MaterialQuantity'])
    AdditiveCatalyst = float(request.form['AdditiveCatalyst'])
    AshComponent = float(request.form['AshComponent'])
    Water = float(request.form['Water'])
    Plasticizer = float(request.form['Plasticizer'])
    ModerateAggregator = float(request.form['ModerateAggregator'])
    RefinedAggregator = float(request.form['RefinedAggregator'])
    FormulationDuration = float(request.form['FormulationDuration'])


    features_for_prediction = np.array([[
        MaterialQuantity, AdditiveCatalyst, AshComponent, Water,
        Plasticizer, ModerateAggregator, RefinedAggregator, FormulationDuration
    ]])


    prediction = model.predict(features_for_prediction)

    return render_template('index.html', strength=prediction[0][0])

if __name__ == "__main__":
    app.run(debug=True)
