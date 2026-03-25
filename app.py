from flask import Flask, request, render_template_string
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load model
model = joblib.load("model.pkl")

# HTML UI (Single Page)
html_page = """
<!DOCTYPE html>
<html>
<head>
    <title>House Price Predictor</title>
    <style>
        body {
            font-family: Arial;
            text-align: center;
            margin-top: 50px;
            background-color: #f2f2f2;
        }
        input {
            padding: 8px;
            margin: 5px;
        }
        button {
            padding: 10px;
            background-color: green;
            color: white;
            border: none;
        }
        .result {
            margin-top: 20px;
            font-size: 20px;
            color: blue;
        }
    </style>
</head>
<body>

    <h2>🏠 House Price Prediction</h2>

    <form method="POST">
        <input type="number" name="area" placeholder="Area (sq ft)" required><br>
        <input type="number" name="bedrooms" placeholder="Bedrooms" required><br>
        <input type="number" name="bathrooms" placeholder="Bathrooms" required><br>
        <input type="number" name="stories" placeholder="Stories" required><br>
        <button type="submit">Predict</button>
    </form>

    {% if prediction %}
    <div class="result">
        {{ prediction }}
    </div>
    {% endif %}

</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        try:
            area = float(request.form["area"])
            bedrooms = float(request.form["bedrooms"])
            bathrooms = float(request.form["bathrooms"])
            stories = float(request.form["stories"])

            features = np.array([[area, bedrooms, bathrooms, stories]])
            result = model.predict(features)[0]

            prediction = f"Predicted Price: ₹ {round(result, 2)}"

        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template_string(html_page, prediction=prediction)

# IMPORTANT for Render
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)