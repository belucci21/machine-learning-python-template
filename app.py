from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import numpy as np

# Inicializar Flask
app = Flask(__name__)

# Cargar el modelo guardado
model_path = "/workspace/machine-learning-python-template/models/best_rf_model.pkl"
with open(model_path, "rb") as f:
    model = pickle.load(f)

# Ruta principal para la página web
@app.route("/")
def home():
    return render_template("index.html")

# Ruta para hacer predicciones
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()  # Obtener datos en formato JSON
        input_data = pd.DataFrame([data])  # Convertir a DataFrame

        # Realizar predicción
        prediction = model.predict(input_data)[0]

        return jsonify({"prediction": int(prediction)})

    except Exception as e:
        return jsonify({"error": str(e)})

# Ejecutar la aplicación
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
