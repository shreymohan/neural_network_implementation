# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 15:31:21 2017

@author: shrey
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
import pandas as pd
import math

def train__nn(X,Y,hidden_num,output_num,epochs=100,alpha=0.01):
    input_num=X.shape[0] #number of input units
    m=X.shape[1] #number of samples
    cost=[]
    #initialize weights
    weights1=np.random.uniform(low=0.0,high=1.0,size=(hidden_num,input_num+1))
    weights2=np.random.uniform(low=0.0,high=1.0,size=(output_num,hidden_num+1))
    for i in range(epochs):    
        j=0  #cost of every sample
        ddw1=0  #weight adustment for every sample
        ddw2=0
        for n in range(m):
            x=X[:,n]
            y=Y[n] 
            #forward prop
            X1=np.append([1],x) #add bias
            X1=X1.reshape((len(x)+1,1))
            
            Z1=np.dot(weights1,X1)
            a1=1 / (1 + np.exp(-Z1))
            A1=np.append([1],a1)
            A1=A1.reshape((len(A1),1))
        
            Z2=np.dot(weights2,A1)

            A2=1 / (1 + np.exp(-Z2))   

            j+=-((y*math.log(A2))+((1-y)*math.log(1-A2))) #add cost of every sample
        
            #back prop
            #for l=2
            dj=A2-y
            ddw2+=np.dot(dj,A1.reshape((1,hidden_num+1))) # add weight adjustments for all samples
        
            #for l=1
            da1=np.multiply(a1,(1-a1))
            theta_w=weights2[:,1:]
            djda=np.multiply(dj,theta_w.T)
            djdz=np.multiply(djda,da1)
            ddw1+=np.dot(djdz,X1.T) # add weight adjustments for all samples
        
        dw2=ddw2/m 
        dw1=ddw1/m
        c=j/m 
        cost.append(c)
        #update weights
        weights1=weights1-alpha*dw1
        weights2=weights2-alpha*dw2
        #print('loss at epoch '+str(i)+' is '+str(c))
    res={'weight1':weights1,'weight2':weights2,'cost':cost}
    return res
       
def nn_predict(X,res):
    weights1=res['weight1']
    weights2=res['weight2']
    
    X1=np.append([1],X)
    X1=X1.reshape((len(X)+1,1))
            
    Z1=np.dot(weights1,X1)
    a1=1 / (1 + np.exp(-Z1))
    A1=np.append([1],a1)
    A1=A1.reshape((len(A1),1))
        
    Z2=np.dot(weights2,A1)

    A2=1 / (1 + np.exp(-Z2))
    A2=A2>0.5
    A2=A2.astype('float')
    return A2
    
# load iris dataset        
iris = datasets.load_iris()
X = iris.data
y = iris.target

# adding 4 extra columns for normalization
df=pd.DataFrame(X,columns=['A','B','C','D'])
df['E']=(df['A']-df['A'].min())/(df['A'].max()-df['A'].min())
df['F']=(df['B']-df['B'].min())/(df['B'].max()-df['B'].min())
df['G']=(df['C']-df['C'].min())/(df['C'].max()-df['C'].min())
df['H']=(df['D']-df['D'].min())/(df['D'].max()-df['D'].min())

data=np.array(df) # convert back to array

X=data[:,6:] # taking petal lengths and petal widths
X=X[50:]  # removing setosa
y=y[50:]
y[y==1]=0  # 0 for versicolor
y[y==2]=1  # 1 for virginica
X=X.T

# Leave one out analysis
total_samples=X.shape[1]
error=0
 
for i in range(total_samples):
    test_x=X[:,i]
    test_y=y[i]
    train_x=np.delete(X,i,axis=1)
    train_y=np.delete(y,i)
    res=train__nn(train_x,train_y,2,1,1200,0.4)
    pre_y=nn_predict(test_x,res)
    if pre_y==test_y:
        error+=0
    else:
        error+=1
        
Error=float(error)/float(total_samples)        
print('total error after leave one out analysis '+str(Error*100)+'%')        
    
    
        
    
    






