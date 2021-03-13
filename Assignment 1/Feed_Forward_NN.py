import numpy as np
from activation_fns_and_grads import *

class FeedForward_NN:
    def __init__(self, input_features, output_nodes, hidden_layers_dims=[2], act_fn='relu', initialization='xavier', dropout=1):

        self.hL = len(hidden_layers_dims)
        self.output_nodes = output_nodes
        self.layer_dims = [input_features] + hidden_layers_dims + [output_nodes] # list with all layer sizes (including input and output)
        
        self.params = {}
        L = len(self.layer_dims) # total number of layers in the network
        
        self.act_fn=act_fn
        ## Weight Initialization
        if initialization=='random':
            for l in range(1, L):

                self.params['W' + str(l)] = np.random.randn(self.layer_dims[l], self.layer_dims[l-1]) * 0.01
                self.params['b' + str(l)] = np.zeros((self.layer_dims[l], 1))
                
        elif initialization=='xavier':
            for l in range(1, L):
                self.params["W" + str(l)] = 0.01*np.random.randn(self.layer_dims[l], self.layer_dims[l-1])*(np.sqrt(2/(self.layer_dims[l] + self.layer_dims[l-1])))
                self.params['b' + str(l)] = np.zeros((self.layer_dims[l], 1))
        
        self.L = len(self.params) // 2 # number of layers in the neural network
        self.caches=[] # stores tuple of X_W (W, X) and activation cache for each layer
        self.grads = {} # it is storing all the partial derivatives i.e. dX, db and dW

        self.dropout=dropout
        if self.dropout != 1:
            self.d_cache={}
    
    def forward_prop(self, X):
     
        self.caches.clear() # self.caches stores activation cache for each layer including input
        
        for l in range(1, self.L):
            W = self.params['W%d' % l]
            b = self.params['b%d' % l]
            
            self.caches.append(X)
            X = np.dot(W , X) + b
            A = act_fn(X,self.act_fn)
            if self.dropout !=1:
                D = np.random.rand(A.shape[0], A.shape[1])
                D = D < self.dropout
                A = A * D
                A = A / self.dropout
                self.d_cache['D%d' % l] = D
            X = A # current activation will act as input for next layer
            
        self.caches.append(X)
        
        W = self.params['W%d' % self.L]
        b = self.params['b%d' % self.L]
        ## last layer activation will be done in train function
        X = np.dot(W , X) + b
        return X
    
    
    def backward_prop(self,dX): # dX is derivaltive of loss fn wrt last layer X i.e, X = W_L @ A_L-1 + b_L
        self.grads.clear()
        self.m = dX.shape[1]
        
        # Now you know partial derivative of last layer activation wrt loss, calculate the weights, biases and L-1 layer activation gradients 
        A_prev = self.caches[self.L-1] # caches stores activations of all layer(last layer not required) + input
        
        W = self.params["W"+str(self.L)]
        
        self.grads["dW" + str(self.L)] = (1 / self.m) * np.dot(dX , A_prev.T)
        self.grads["db" + str(self.L)] = (1 / self.m) * np.sum(dX, axis=1, keepdims=True)
        self.grads["dA" + str(self.L-1)] = np.dot(W.T , dX)
        
        for l in reversed(range(self.L-1)):
            if self.dropout != 1:
                self.grads["dA" + str(l+1)] = self.grads["dA" + str(l+1)] * self.d_cache['D' + str(l+1)]
                self.grads["dA" + str(l+1)] = self.grads["dA" + str(l+1)] / self.dropout
            W = self.params["W"+str(l+1)]
            
            dX = act_fn_grad(self.grads["dA" + str(l + 1)], A_prev, self.act_fn) # in first iteration it is dx_L-1
            A_prev = self.caches[l]                                                                # use dX_L-1 to find dW_L-1, db_L-1 and dA_L-2
            self.grads["dW" + str(l+1)] = (1 / self.m) * np.dot(dX , A_prev.T)
            self.grads["db" + str(l+1)] = (1 / self.m) * np.sum(dX, axis=1, keepdims=True)
            self.grads["dA" + str(l)] = np.dot(W.T , dX)
            
