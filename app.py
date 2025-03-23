import numpy as np
import pickle
import math
from flask import Flask,request,jsonify,render_template

app = Flask(__name__,template_folder='template',static_folder='staticfiles') ##assign flask = app
model = pickle.loan(open('build.pkl','rb')) #import model

@app.route('/')
def home():
    return render_template('index.html') ##reading index.html file

@app.route('/predict',methods=['POST'])  ##tranfer data from html to python/server  

def predict():
    int_features= [int(x) for x in request.form.values()] #request for data values
    final_features= [np.array(int_features)] #convert into array
    prediction = model.predict(final_features) #predict
    if prediction == 1:
        return render_template('index.html',prediction_text='loan is rejected').format(prediction)
    else:
        return render_template('index.html',prediction_text='loan is approved').format(prediction)

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8080)
