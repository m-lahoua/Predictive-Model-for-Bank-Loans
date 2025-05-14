import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Liste des noms de caractéristiques
    features = ['Credit_History', 'Married', 'CoapplicantIncome']

    # Initialisez les caractéristiques à zéro par défaut
    input_features = [0] * len(features)

    # Mettez à jour les caractéristiques en fonction des valeurs de la requête
    for i, feature in enumerate(features):
        if request.form.get(feature) == '1':
            input_features[i] = 1

    # Effectuez la prédiction avec les caractéristiques traitées
    prediction = model.predict([np.array(input_features)])

    output = 'Approuvé' if prediction[0] == 1 else 'Non approuvé'

    return render_template('index.html', prediction_text='Décision du crédit : {}'.format(output))

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    features = ['Credit_History', 'Married', 'CoapplicantIncome']

    input_features = [0] * len(features)
    for i, feature in enumerate(features):
        if data.get(feature) == 1:
            input_features[i] = 1

    prediction = model.predict([np.array(input_features)])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)
