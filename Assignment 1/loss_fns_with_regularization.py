import numpy as np
#from utils import *

# Loss Functions

## Cross entropy for softmax activation. Here no of classes is equal to output nodes
def cross_entropy_softmax(y,y_pred,already_one_hot=False,zero_indexed_classes=True):
    try:
        m = y_pred.shape[1]
    except:
        m=1
    classes = y_pred.shape[0]
    if already_one_hot:
        return np.sum(np.array(y)*(-1*np.log(y_pred))) / m
    else:
        # classes should begin with zero index or else change one_hot_fn
        return np.sum(class_onehot(y , classes, zero_indexed=zero_indexed_classes)*(-1*np.log(y_pred))) / m

def cross_entropy_softmax_grad(y,y_pred,already_one_hot=False,zero_indexed_classes=True):
    try:
        m = y_pred.shape[1]
    except:
        m=1
    classes = y_pred.shape[0]
    if already_one_hot:
        return np.array(y)*(-1/y_pred)*(1/m)
    else:
        # classes should begin with zero index or else change one_hot_fn
        return class_onehot(y,classes,zero_indexed=zero_indexed_classes)*(-1/y_pred)*(1/m)

def l2_regularizer_cost(params,lambd):
    L = len(params)//2
    L2_regularization_cost=0
    for l in range(L):
        L2_regularization_cost += np.sum(np.square(params['W' + str(l+1)]))
    L2_regularization_cost = lambd* L2_regularization_cost

    return L2_regularization_cost

## Mean squared loss
def mean_squared(y, y_pred):
    try:
        m = y_pred.shape[1]
    except:
        m=1
    
    return  np.sum((y_pred - y)**2 ).squeeze() / (2*m)

def mean_squared_grad(y, y_pred):
    try:
        m = y_pred.shape[1]
    except:
        m=1

    return (y_pred - y) / m


## Cross entropy for sigmoid activaation. Here we have single output node
def cross_entropy_sigmoid(y, y_pred): # cross entropy sigmoid is for single output node with sigmoid activation
    try:
        m = y_pred.shape[1]
    except:
        m=1
    
    return  -np.sum(y * np.log(y_pred) + (1-y) * np.log(1-y_pred)).squeeze() / m

def cross_entropy_sigmoid_grad(y, y_pred):
    try:
        m = y_pred.shape[1]
    except:
        m=1
    
    return -(np.divide(y, y_pred) - np.divide(1 - y, 1 - y_pred)) / m

# Access to each loss function

def cost_fn(params,y,y_pred,l2_lambd=0, output_activation="linear", loss_fn="cross_entropy"):
    try:
        m = y_pred.shape[1]
    except:
        m=1
    if output_activation=="sigmoid" and loss_fn=="cross_entropy":
        actual_cost = cross_entropy_sigmoid(y, y_pred)
    elif output_activation=="softmax" and loss_fn=="cross_entropy":
        actual_cost = cross_entropy_softmax(y,y_pred)
    elif output_activation=="linear" and loss_fn=="mse":
        actual_cost = mean_squared(y, y_pred)

    if l2_lambd != 0:
        return actual_cost + l2_regularizer_cost(params,l2_lambd)/(2*m)

    else:
        return actual_cost

    
## Class to one hot
def class_onehot(indices, num_classes , zero_indexed=True): # Takes class number [0,classes-1] and returns one hot (zero indexed)
    #print(indices)
    indices = indices.astype('int') 
    if zero_indexed:
        return np.identity(num_classes)[np.array(indices)].T.squeeze()
    else :
        return np.identity(num_classes)[np.array(indices-1)].T.squeeze()
