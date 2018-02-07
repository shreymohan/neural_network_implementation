# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 18:09:09 2017

@author: shrey
"""

import numpy as np
import matplotlib.pyplot as plt

def train_nn(x,y,hidden_num,output_num,epochs=100,alpha=0.01):
    input_num=x.shape[0]
    m=x.shape[1]
    weights1=np.random.uniform(low=0.0,high=1.0,size=(hidden_num,input_num+1))
    weights2=np.random.uniform(low=0.0,high=1.0,size=(output_num,hidden_num+1))
    cost=[]
  
    for i in range(epochs):
        #forward prop
        X=np.append([1],x)
        X=X.reshape((len(x)+1,1))

        Z1=np.dot(weights1,X)
        a1=1 / (1 + np.exp(-Z1))
        A1=np.append([1],a1)
        A1=A1.reshape((len(A1),1))

        Z2=np.dot(weights2,A1)

        A2=1 / (1 + np.exp(-Z2))

        j=(y-A2)**2/2*m

        J_total=j.sum()

        #back prop
        #for l=2, the output layer
        dj=A2-y
        da2=np.multiply(A2,(1-A2))
        dz2=np.multiply(dj,da2)
        dw2=np.dot(dz2,A1.reshape((1,hidden_num+1)))

        #for l=1, the hidden layer
        da1=np.multiply(a1,(1-a1))
        theta_w=weights2[:,1:]
        dz1=np.dot(dz2.T,theta_w)
        djdz=np.multiply(dz1.T,da1)
        dw1=np.dot(djdz,X.T)
                
        #update weights
        weights1=weights1-alpha*dw1
        weights2=weights2-alpha*dw2
        
        #keep track of changing cost and thetas
        cost.append(J_total)
  
        print('loss at epoch '+str(i)+' is '+str(J_total))
        
    res={'weight1':weights1,'weight2':weights2,'cost':cost}
    return res      
    
def predict_nn(x,res):
     weights1=res['weight1']
     weights2=res['weight2']
     X=np.append([1],x)
     X=X.reshape((len(x)+1,1))
     Z1=np.dot(weights1,X)
     a1=1 / (1 + np.exp(-Z1))
     A1=np.append([1],a1)
     A1=A1.reshape((len(A1),1))

     Z2=np.dot(weights2,A1)

     A2=1 / (1 + np.exp(-Z2))
     return A2
        


x=np.array([[0.05],[0.1]])
y=np.array([[0.01],[0.99]])

res=train_nn(x,y,2,2,400,0.04)

plt.plot(res['cost'])

test_val=np.array([[0.04],[0.12]])

test_res=predict_nn(test_val,res)















