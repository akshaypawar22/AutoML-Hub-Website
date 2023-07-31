import sklearn
import numpy as np
from flask import Flask, request, jsonify, render_template, redirect
import pickle
import pandas as pd
from flask_cors import CORS, cross_origin
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler


sc = StandardScaler()

app = Flask(__name__)
cors = CORS(app)


# print(sklearn.__version__+" Hello ")

modelFraud = pickle.load(open('modelNewsvc.pkl', 'rb'))
modelPrice = pickle.load(open('LinearRegressionModelnew.pkl', 'rb'))
car = pd.read_csv('Cleaned_Car_data.csv')

# Home page API


@app.route('/')
def home():
    return render_template('index.html')

# Auto insurance fraud Detection API


@app.route('/fraud')
def fraud():
    return render_template('fraud.html')


@app.route('/predictFraud', methods=['POST'])
def predictFraud():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    # print(final_features)
    prediction = modelFraud.predict(final_features)

    output = round(prediction[0], 2)
    result = ""
    if (output == 1):
        result = "Fraud"
    else:
        result = "Not Fraud"

    return render_template('fraud.html', prediction_text='Above Insurance Case is '+result)


@app.route('/predict_api', methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = modelFraud.predict([np.array(list(data.values()))])
    output = prediction[0]
    return jsonify(output)

# Car Price prediction API


@app.route('/CarPrice', methods=['GET', 'POST'])
def index():
    companies = sorted(car['company'].unique())
    car_models = sorted(car['name'].unique())
    year = sorted(car['year'].unique(), reverse=True)
    fuel_type = car['fuel_type'].unique()
    companies.insert(0, 'Select Company')
    return render_template('price.html', companies=companies, car_models=car_models, years=year, fuel_types=fuel_type)


@app.route('/predictPrice', methods=['POST'])
@cross_origin()
def predictPrice():

    company = request.form.get('company')
    car_model = request.form.get('car_models')
    year = request.form.get('year')
    fuel_type = request.form.get('fuel_type')
    driven = request.form.get('kilo_driven')

    prediction = modelPrice.predict(pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],
                                                 data=np.array([car_model, company, year, driven, fuel_type]).reshape(1, 5)))
    # print(prediction)

    return str(np.round(prediction[0], 2))


# Car loan status prediction API
model = pickle.load(open('NewRFmodel.pkl', 'rb'))


@app.route('/CarLoan')
def loan():
    return render_template('CarLoan.html')


@app.route('/predictLoan', methods=['POST'])
def predict():

    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)
    predtext = ""
    if (output == 1):
        predtext = "YOUR CAR LOAN MAY GET APPROVED"
    else:
        predtext = "YOUR CAR LOAN MAY NOT GET APPROVED"

    return render_template('CarLoan.html', prediction_text=predtext)


@app.route('/results', methods=['POST'])
def results():

    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

# Fuel Efficiency Prediction API


@app.route("/Fuel")
def Fuel():
    return render_template('Fuel.html')


@app.route('/Fuelresult', methods=['POST', 'GET'])
def FuelResult():
    cylinders = int(request.form["cylinders"])
    displacement = int(request.form["displacement"])
    horsepower = int(request.form["horsepower"])
    weight = int(request.form["weight"])
    acceleration = int(request.form["acceleration"])
    model_year = int(request.form["model_year"])
    origin = int(request.form["origin"])

    values = [[cylinders, displacement, horsepower,
               weight, acceleration, model_year, origin]]

    sc = None
    with open('scaler.pkl', 'rb') as f:
        sc = pickle.load(f)

    values = sc.transform(values)
    modelFuel = load_model("model.h5")

    prediction = modelFuel.predict(values)
    prediction = float(prediction)
    if(prediction >= 50.0):
        prediction = 0.0
    # print(prediction)

    return render_template('Fuel.html', FuelPrediction_text="The Estimated Engine's miles per gallon is {}".format(prediction))


@app.route('/About')
def About():
    return render_template('About.html')


@app.route('/Blogs')
def blogs():
    return render_template('blogs.html')


@app.route('/contact')
def contact():
    return render_template('contact.html')


if __name__ == "__main__":
    app.run(debug=True)
