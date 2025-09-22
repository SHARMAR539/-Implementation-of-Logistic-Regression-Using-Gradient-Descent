# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Load dataset and remove unwanted columns.
2.Convert categorical variables into numeric codes.
3.Split data into features X and target Y.
4.Initialize weight vector theta randomly.
5.Define sigmoid function for probability calculation.
6.Apply gradient descent to update theta.
7.Predict outcomes using sigmoid and threshold 0.5.
8.Evaluate accuracy and test with new input samples.
```
## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Sharma R
RegisterNumber:  212224230261
*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset=pd.read_csv("Placement_Data.csv")
print(dataset)

dataset = dataset.drop('sl_no',axis=1)

dataset =dataset.drop('salary',axis=1)

dataset["gender"] = dataset["gender"].astype('category')
dataset["ssc_b"] = dataset["ssc_b"].astype('category')
dataset["hsc_b"] = dataset["hsc_b"].astype('category')
dataset["degree_t"] = dataset["degree_t"].astype('category')
dataset["workex"] = dataset["workex"].astype('category')
dataset["specialisation"] = dataset["specialisation"].astype('category')
dataset["status"] = dataset["status"].astype('category')
dataset["hsc_s"] = dataset["hsc_s"].astype('category')
print(dataset.dtypes)

dataset["gender"] = dataset["gender"].cat.codes
dataset["ssc_b"] = dataset["ssc_b"].cat.codes
dataset["hsc_b"] = dataset["hsc_b"].cat.codes
dataset["degree_t"] = dataset["degree_t"].cat.codes
dataset["workex"] = dataset["workex"].cat.codes
dataset["specialisation"] = dataset["specialisation"].cat.codes
dataset["status"] = dataset["status"].cat.codes
dataset["hsc_s"] = dataset["hsc_s"].cat.codes

print(dataset)

X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

print(Y)

theta = np.random.randn(X.shape[1])
y=Y

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def loss(theta,X,y):
    h=sigmoid(X.dot(theta))
    return -np.sum(y * np.log(h) + (1-y) * np.log(1-h))

def gradient_descent(theta, X, y, alpha, num_iterations):
    m=len(y)
    for i in range (num_iterations):
        h=sigmoid(X.dot(theta))
        gradient=X.T.dot(h-y) / m
        theta -= alpha * gradient
    return theta

theta = gradient_descent(theta, X, y, alpha=0.01, num_iterations=1000)

def predict(theta, X):
    h = sigmoid(X.dot(theta))
    y_pred = np.where(h >= 0.5,1,0)
    return y_pred

y_pred = predict(theta, X)

accuracy = np.mean(y_pred.flatten()==y)
print("Accuracy:",accuracy)

print(y_pred)

print(Y)

xnew=np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)

xnew=np.array([[0,0,0,0,0,2,8,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew) 
```

## Output:
![ml 1](https://github.com/pragalyaashree/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/128135934/f9de7c52-dbb0-495d-ac78-751516f40ab5)
![ml 3](https://github.com/pragalyaashree/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/128135934/45b1b15c-22b0-40f6-a370-70f1d6e778d8)
![ml 4](https://github.com/pragalyaashree/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/128135934/fe0fcf43-fe6f-49db-8795-0443d9db519c)
![ml 5](https://github.com/pragalyaashree/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/128135934/d4c68e66-238b-4835-b13f-71329112d9db)
![ml 6](https://github.com/pragalyaashree/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/128135934/28604ede-b3fa-496b-808b-cce2962f3961)
![ml 8](https://github.com/pragalyaashree/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/128135934/370c86a1-e9b2-4c69-b6ee-75c74f056eb8)


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

