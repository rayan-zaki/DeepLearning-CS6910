import numpy as np

# Activation Functions
def sigmoid(X):
    A = 1/(1+np.exp(-X))
    return A

def tanh(X):
    A = (np.exp(X)-np.exp(-X))/(np.exp(X)+np.exp(-X))
    return A

def tanh_grad(dA,A):
    dX = dA * (1 - A**2)
    return dX

def relu(X):
    A = np.maximum(0,X)
    return A

def relu_grad(dA,A):
    dX = np.array(dA, copy=True)
    dX[A<=0] = 0
    return dX

def sigmoid_grad(dA,A):
    dX = dA * A* (1-A)
    return dX

def softmax(X):
    A=np.exp(X)
    A = A/np.sum(A, axis=0, keepdims=True)
    return A

def act_fn(X,activation):
    if activation == "sigmoid":
        return sigmoid(X)

    elif activation == "tanh":
        return tanh(X)

    elif activation == "relu":
        return relu(X)

    elif activation == "softmax":
        return softmax(X)
    else:
        print("Please include the fn first")
            
def act_fn_grad(dA, A, activation_name):
    # Returns dX
    if activation_name == "relu":
        return relu_grad(dA, A)

    elif activation_name == "sigmoid":
        return sigmoid_grad(dA, A)

    elif activation_name == "tanh":
        return tanh_grad(dA, A)
