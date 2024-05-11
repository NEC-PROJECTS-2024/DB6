import joblib
import numpy as np
import pandas as pd
train_x_std = pd.read_csv('train_x_std.csv')
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=2)
poly_reg.fit(train_x_std)
BR=joblib.load('model.pkl')

from flask import Flask, request, render_template

# Create a Flask application instance
app = Flask(__name__)
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        age = float(request.form['age'])
        phd = int(request.form['phd'])
        gender = int(request.form['gender'])

        # Create a feature array based on user input
        user_input = np.array([[age, phd, gender]])
        user_input_poly = poly_reg.transform(user_input)

        # Get the salary prediction for the user input
        salary_prediction = BR.predict(user_input_poly)[0]

        return render_template('result.html', salary=salary_prediction)
    
    return render_template('index.html')