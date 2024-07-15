import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import json as JS

Data = pd.read_csv("Breast_Canceer_Data.csv")

e = math.e
Features = 6
Parameters = []
TestOutput = []
TestInput = []
L = 0.9

def InitParam(Parameters,Features):
    for i in range(Features + 1):
        Parameters.append(0)
    return Parameters

def Input(TestInput,Data):
    for i in range(len(Data)):
        Radius = Data.iloc[i].radius_mean/100
        Texture = Data.iloc[i].texture_mean/100
        Perimeter = Data.iloc[i].perimeter_mean/1000
        Smoothness = Data.iloc[i].smoothness_mean
        Area = Data.iloc[i].area_mean/10000
        Compactness = Data.iloc[i].compactness_mean
        Concave = Data.iloc[i].concave_points_mean
        Concavity = Data.iloc[i].concavity_mean
        Symmetry = Data.iloc[i].symmetry_mean
        FractalDim = Data.iloc[i].fractal_dimension_mean
        TestInput.append([Radius,Texture,Perimeter,Smoothness,Area,Compactness,Concave,Concavity,Symmetry,FractalDim])
    return TestInput

def Output(TestOutput,Data):
    for i in range(len(Data)):
        Diagnosis = Data.iloc[i].diagnosis
        if Diagnosis == 'M':
            Diagnosis = 1
        else:
            Diagnosis = 0
        
        TestOutput.append(Diagnosis)
    return TestOutput


def Sigmoid(z):
    return 1/(1+ e ** -z)

def Hypothesis(Parameters,X):
    value = 0
    for i in range(len(Parameters)):
        if i == 0:
            value = Parameters[0]
        else:
            value += Parameters[i] * X[i-1]
    return value

def Prediction(Parameters,TestInput,TestOutput,L):
    for j in range(len(Parameters)):
        Error = 0
        for i in range(len(Data)):
            X = TestInput[i]
            Y = TestOutput[i]
            z = Hypothesis(Parameters,X)
            hx = Sigmoid(z)
            if i == 0:
                Error = (Y - hx)
            else:
                Error += (Y - hx) * X[j-1]
        Parameters[j] += L * Error
    return Parameters

def Result(Parameters,X):
    z = Hypothesis(Parameters,X)
    hx = Sigmoid(z)
    # if hx > 0.5:
    #     return 'M'
    # else:
    #     return 'B'
    return round(hx,20)

def MeanSquaredError(Parameters,TestInput,TestOutput):
    Error = 0
    for i in range(len(Data)):
        X = TestInput[i]
        Y = TestOutput[i]
        z = Hypothesis(Parameters,X)
        hx = Sigmoid(z)
        Error += (1/2) * (Y - hx) ** 2
    return Error

TestOutput = Output(TestOutput,Data)
TestInput = Input(TestInput,Data)

if __name__ == "__main__":
    print(f"In Main of LogsiticRegression")
    # while True:
    #     with open('BreastCancer.json') as JSfile:
    #         data = JS.load(JSfile)
    #         Parameters = data[0]
    #         # Error = data[1]

    #     for i in range(1000):
    #         Parameters = Prediction(Parameters,TestInput,TestOutput,L)
        
    #     Error = MeanSquaredError(Parameters,TestInput,TestOutput)
    #     print(f"Error: {Error}") 
    #     print(Parameters)
    #     print("\n")
    #     with open('BreastCancer.json','w') as file:
    #         JS.dump([Parameters,Error],file)


