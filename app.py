from flask import Flask, request, jsonify, render_template
import pickle
import os
import pandas as pd

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
        input_data = pd.DataFrame([data], columns=model.feature_names_in_)

        prediction = model.predict(input_data)[0]
        return jsonify({"prediction": prediction})

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    print("Iniciando Flask en el puerto 5000...")
    app.run(host="0.0.0.0", port=5000, debug=True)
