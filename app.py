import numpy as np
from flask import Flask, render_template, request, redirect
from flask_cors import CORS, cross_origin
import pickle
import pandas as pd
from keras.models import load_model

app=Flask(__name__)

cors=CORS(app)
model = load_model('model.h5')

@app.route('/',methods=['GET','POST'])
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
@cross_origin()
def predict():

    veg = request.form.get('veg')
    season =request.form.get('season')
    temp = request.form.get('temp')
    mon = request.form.get('mon')
    condition = request.form.get('condition')
    disaster = request.form.get('disaster')
    
    prediction = model.predict([[int(veg),int(season),int(temp),
                              int(mon),int(condition),int(disaster)]])
    print(prediction)

    return str(np.round(prediction[0][0],2))
