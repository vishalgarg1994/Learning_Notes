from flask import Flask, request
import numpy as np
import pickle
import pandas as pd
import flasgger
from flasgger import Swagger

app=Flask(__name__)
Swagger(app)

pickle_in = open("model.pkl","rb")
regressor=pickle.load(pickle_in)

@app.route('/')
def welcome():
    return "Welcome All"

@app.route('/predict',methods=["Get"])
def predict_value():
    
    """Let's get value based on test Value
    This is using docstrings for specifications.
    ---
    parameters:  
      - name: value
        in: query
        type: number
        required: true
    responses:
        200:
            description: The output values
        
    """
    value=request.args.get("value")
    value = np.array([[int(value)]])
    prediction=regressor.predict(value)
    print(prediction)
    return "Hello The answer is"+str(prediction)

@app.route('/predict_file',methods=["POST"])
def predict_value_file():
    """Let's get Salary based on experience2
    This is using docstrings for specifications.
    ---
    parameters:
      - name: value
        in: formData
        type: file
        required: true
      
    responses:
        200:
            description: The output values
        
    """
    df_test=pd.read_csv(request.files.get("value"))
    print(df_test.head())
    prediction=regressor.predict(df_test)
    return str(list(prediction))


if __name__=='__main__':
    app.run()