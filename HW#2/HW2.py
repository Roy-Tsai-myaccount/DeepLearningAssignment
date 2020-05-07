# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os

path = os.getcwd() +'\ex1data1.txt'
data = pd.read_csv(path,header=None,names=['Population','Profit'])
data.head()

data.describe()

data.plot(kind='scatter',x='Population',y='Profit',figsize=(12,8))


#C:\Users\s2102\AppData\Roaming\SPB_Data\.spyder-py3\Data
def computeCost(X,y,theta):
    inner = np.power(((X * theta.T)-y),2)
    return np.sum(inner)/(2 * len(X))

data.insert(0,'Ones',1)

cols = data.shape[1]
X = data.iloc[:,0:cols-1]
y = data.iloc[:,cols-1:cols]

X = np.matrix(X.values)
y = np.matrix(y.values)
theta = np.matrix(np.array([0,0]))

def gradientDescent(X,y,theta,alpha,iters):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iters)
    
    for i in range(iters):
        error = (X * theta.T) - y
        
        for j in range(parameters):
            term = np.multiply(error,X[:,j])
            temp[0,j] = theta[0,j]-((alpha/len(X))*np.sum(term))
            
        theta = temp
        cost[i]=computeCost(X,y,theta)
            

    return theta,cost

alpha = 0.01
iters = 5000

g, cost = gradientDescent(X,y,theta,alpha,iters)
print('iters = ', iters ,'cost =  ' ,cost[iters-1])


x = np.linspace(data.Population.min(),data.Population.max(),100)
f = g[0,0] + (g[0,1] * x)

'''
fig,ax =plt.subplots(figsize = (12,8))
ax.plot(x,f,'r',label='Prediction')
ax.scatter(data.Population,data.Profit,label = 'Traning Data')
ax.legend(loc = 2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')

fig,ax = plt.subplots(figsize=(12,8))
ax.plot(np.arange(iters),cost,'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')
'''
'''    
from sklearn import linear_model
model = linear_model.LinearRegression()
model.fit(X,y)

x =np.array(X[:,1].A1)
f = model.predict(X).flatten()

fig,ax = plt.subplots(figsize=(12,8))
ax.plot(x,f,'r',label='Prediction')
ax.scatter(data.Population,data.Profit,label = 'Traning Data')
ax.legend(loc = 2)
ax.set_xlabel('Population')
ax.set_ylabel('Profit')
ax.set_title('Predicted Profit vs. Population Size')
'''