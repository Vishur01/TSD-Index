from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__, template_folder='template')



model = pickle.load(open('TSDgbr.pkl', 'rb'))

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        try:
            # Get 8 inputs from form
            inputs = [
                float(request.form[f"input{i}"])
                for i in range(1, 9)
            ]

            # Convert to numpy array
            final_input = np.array(inputs).reshape(1, -1)

            # Predict
            prediction = model.predict(final_input)[0]

        except Exception as e:
            prediction = f"Error: {e}"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
