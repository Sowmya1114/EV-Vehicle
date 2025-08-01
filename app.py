from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load your trained model
model = joblib.load('ev_adoption_forecast_model.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        try:
            # Get inputs from form
            year = int(request.form['year'])
            month = int(request.form['month'])
            percent_ev = float(request.form['percent_ev'])
            county_code = int(request.form['county_code'])
            state_code = int(request.form['state_code'])

            # Prepare data for prediction
            features = np.array([[year, month, percent_ev, county_code, state_code]])

            # Predict
            prediction = model.predict(features)[0]
            prediction = round(prediction, 2)
        except Exception as e:
            prediction = f"Error: {str(e)}"

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
