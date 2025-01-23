from flask import Flask, request, jsonify, render_template
import pickle
import os
import pandas as pd
import shap
import numpy as np

app = Flask(__name__)

# Cargar el modelo guardado
model_path = os.path.join(os.getcwd(), "models", "best_rf_model.pkl")

try:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    print("Modelo cargado correctamente.")
except FileNotFoundError:
    print(f"Error: No se encontró el modelo en {model_path}")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        input_data = pd.DataFrame([data])  
        prediction = model.predict(input_data)[0]

        # Calcular valores SHAP para la instancia
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_data)

        # Depuración para ver qué contiene shap_values
        print("Tipo de shap_values:", type(shap_values))
        print("Contenido de shap_values:", shap_values)

        # Obtener los valores SHAP para la clase predicha
        class_index = list(model.classes_).index(prediction)
        shap_values_for_prediction = shap_values[0][:, class_index]

        # Crear un diccionario con los valores SHAP y los nombres de las variables
        shap_dict = dict(zip(model.feature_names_in_, shap_values_for_prediction))

        return jsonify({"prediction": prediction, "shap_values": shap_dict})

    except Exception as e:
        print("Error en predict():", str(e))
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    print("Iniciando Flask en el puerto 5000...")
    app.run(host="0.0.0.0", port=5000, debug=True)
