
from flask import Flask, request, jsonify
from helpers.encoders import *
import pandas as pd
import json
import numpy as np
import joblib 


appw=Flask(__name__)

model = joblib.load('model.hatcars2',"readwrite")
scaler = joblib.load('scaler.hatcars2',"readwrite")


@appw.route('/predict/carprice',methods=['POST'])
def predict():
    car=[]
    car.append(request.json['usedsince'])
    car.append(request.json['closingmirros'])
    car.append(request.json['intellegentparkingsystem'])
    car.append(request.json['sunroof'])
    car.append(request.json['rearcamera'])
    car.append(make_enc[request.json['make']])
    car.append(model_enc[request.json['model']])
    car.append(tran_enc[request.json['transmission']])

    
    car=np.array(car)
    car=scaler.transform([car])
    carprice=model.predict(car)

    return  jsonify({"price": str(carprice).replace('[','').replace(']','')})
if __name__ == "__main__":
    
    appw.run(debug=True)


    







    
