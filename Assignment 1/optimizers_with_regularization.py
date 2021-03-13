import numpy as np


class stochastic_gradient_descent:
    def __init__(self, learning_rate = 0.001, l2_lambd=0):
        self.lr=learning_rate
        self.l2_lambd=l2_lambd
    def update_params(self, model):
        L = len(model.params) // 2
        for l in range(L):
            model.params["W" + str(l+1)] = model.params["W" + str(l+1)] - self.lr * (model.grads["dW" + str(l+1)] +
                                                                                     self.l2_lambd * model.params["W" + str(l+1)] * (1/model.m))
            model.params["b" + str(l+1)] = model.params["b" + str(l+1)] - self.lr * model.grads["db" + str(l+1)]

class momentum_gradient_descent:
    def __init__(self, params, learning_rate = 0.001, gamma = 0.9, l2_lambd=0):
        self.lr=learning_rate
        self.gamma=gamma
        self.l2_lambd=l2_lambd
        
        self.L = len(params) // 2 # number of layers in the neural networks
        self.v = {}

        for l in range(self.L):
            self.v["dW" + str(l+1)] = np.zeros_like(params['W' + str(l+1)])
            self.v["db" + str(l+1)] = np.zeros_like(params['b' + str(l+1)])
        
    def update_params(self, model):
        
        for l in range(self.L):
            
            self.v["dW" + str(l+1)] = self.gamma * self.v["dW" + str(l+1)] + self.lr * (model.grads["dW" + str(l+1)] +
                                                                                        self.l2_lambd * model.params["W" + str(l+1)]* (1/model.m))
            self.v["db" + str(l+1)] = self.gamma * self.v["db" + str(l+1)] + self.lr * model.grads['db' + str(l+1)]
            
            model.params["W" + str(l+1)] = model.params["W" + str(l+1)] - self.v["dW" + str(l+1)] 
            model.params["b" + str(l+1)] = model.params["b" + str(l+1)] - self.v["db" + str(l+1)]

class adam:
    def __init__(self, params, learning_rate = 0.001, beta1= 0.9 ,beta2 = 0.999, l2_lambd=0, epsilon = 1e-8):
        self.lr=learning_rate
        self.beta1=beta1
        self.beta2=beta2
        self.l2_lambd=l2_lambd
        self.epsilon = epsilon
        self.L = len(params) // 2 # number of layers in the neural networks
        self.v = {}
        self.s= {}
        self.t=0;

        for l in range(self.L):
            self.v["dW" + str(l+1)] = np.zeros_like(params['W' + str(l + 1)])
            self.v["db" + str(l+1)] = np.zeros_like(params['b' + str(l + 1)])
            self.s["dW" + str(l+1)] = np.zeros_like(params["W" + str(l + 1)])
            self.s["db" + str(l+1)] = np.zeros_like(params["b" + str(l + 1)])
            
            
    def update_params(self, model):
        v_corrected = {}
        s_corrected = {}

        for l in range(self.L):
                                                                                                                          
            self.t = self.t + 1
            
            self.v["dW" + str(l + 1)] = self.beta1 * self.v["dW" + str(l + 1)] + (1 - self.beta1) * (model.grads["dW" + str(l+1)] +
                                                                                                     self.l2_lambd * model.params["W" + str(l+1)] * (1/model.m) )
            self.v["db" + str(l + 1)] = self.beta1 * self.v["db" + str(l + 1)] + (1 - self.beta1) * (model.grads['db' + str(l + 1)] )
            
            
            v_corrected["dW" + str(l + 1)] = self.v["dW" + str(l + 1)] / (1 - self.beta1**(self.t))#+ self.epsilon)
            v_corrected["db" + str(l + 1)] = self.v["db" + str(l + 1)] / (1 - self.beta1**(self.t))#+ self.epsilon)
            
            
            self.s["dW" + str(l + 1)] = self.beta2 * self.s["dW" + str(l + 1)] + (1 - self.beta2) * (model.grads["dW" + str(l+1)] +
                                                                                                     self.l2_lambd * model.params["W" + str(l+1)]* (1/model.m) )**2
            self.s["db" + str(l + 1)] = self.beta2 * self.s["db" + str(l + 1)] + (1 - self.beta2) * (model.grads['db' + str(l + 1)] )**2
            
            
            s_corrected["dW" + str(l + 1)] = self.s["dW" + str(l + 1)] / (1 - self.beta2**(self.t))# + self.epsilon)
            s_corrected["db" + str(l + 1)] = self.s["db" + str(l + 1)] / (1 - self.beta2**(self.t))# + self.epsilon)
            
                
            model.params["W" + str(l + 1)] = model.params["W" + str(l + 1)] - self.lr * v_corrected["dW" + str(l + 1)] / (np.sqrt(s_corrected["dW" + str(l + 1)]) + self.epsilon)
            model.params["b" + str(l + 1)] = model.params["b" + str(l + 1)] - self.lr * v_corrected["db" + str(l + 1)] / (np.sqrt(s_corrected["db" + str(l + 1)]) + self.epsilon)
            

class rms_prop:
    def __init__(self, params, learning_rate=0.001, beta2 = 0.999, l2_lambd=0, epsilon = 1e-8):
        
        self.lr=learning_rate
        self.beta2=beta2
        self.l2_lambd=l2_lambd
        self.epsilon=epsilon
        self.L = len(params) // 2 # number of layers in the neural networks
        self.s= {}

        for l in range(self.L):
            self.s["dW" + str(l+1)] = np.zeros_like(params["W" + str(l + 1)])
            self.s["db" + str(l+1)] = np.zeros_like(params["b" + str(l + 1)])
            
    def update_params(self, model):

        for l in range(self.L):
            
            self.s["dW" + str(l + 1)] = self.beta2 * self.s["dW" + str(l + 1)] + (1 - self.beta2) * (model.grads["dW" + str(l+1)] +
                                                                                                     self.l2_lambd * model.params["W" + str(l+1)]* (1/model.m) )**2
            self.s["db" + str(l + 1)] = self.beta2 * self.s["db" + str(l + 1)] + (1 - self.beta2) * (model.grads['db' + str(l + 1)] )**2
            
            model.params["W" + str(l + 1)] = model.params["W" + str(l + 1)] - self.lr * ((model.grads["dW" + str(l+1)] +
                                                                                         self.l2_lambd * model.params["W" + str(l+1)]* (1/model.m)) /
                                                                                         (np.sqrt(self.s["dW" + str(l + 1)]) + self.epsilon))
            model.params["b" + str(l + 1)] = model.params["b" + str(l + 1)] - self.lr * ((model.grads['db' + str(l + 1)] ) / (np.sqrt(self.s["db" + str(l + 1)]) +
                                                                                                                                         self.epsilon)) 
            

class nesterov_gradient_descent:
    def __init__(self,params, learning_rate=0.001, gamma =0.9 , l2_lambd=0):
        self.lr=learning_rate
        self.gamma=gamma
        self.l2_lambd=l2_lambd
        self.L = len(params) // 2 # number of layers in the neural networks
        self.v = {}

        for l in range(self.L):
            self.v["dW" + str(l+1)] = np.zeros_like(params['W' + str(l+1)])
            self.v["db" + str(l+1)] = np.zeros_like(params['b' + str(l+1)])
        
    def update_params_prior_grad(self, model):
        for l in range(self.L):
            
            self.v["dW" + str(l+1)] = self.gamma * self.v["dW" + str(l+1)] 
            self.v["db" + str(l+1)] = self.gamma * self.v["db" + str(l+1)] 
            
            model.params["W" + str(l+1)] = model.params["W" + str(l+1)] - self.v["dW" + str(l+1)]
            model.params["b" + str(l+1)] = model.params["b" + str(l+1)] - self.v["db" + str(l+1)]
            
    def update_params_after_grad(self, model):
        for l in range(self.L):

            self.v["dW" + str(l+1)] = self.v["dW" + str(l+1)] + self.lr * (model.grads["dW" + str(l+1)] +
                                                                            self.l2_lambd * model.params["W" + str(l+1)]* (1/model.m)  )
            self.v["db" + str(l+1)] = self.v["db" + str(l+1)] + self.lr * (model.grads['db' + str(l+1)] )
            
            model.params["W" + str(l+1)] = model.params["W" + str(l+1)] - self.lr * (model.grads["dW" + str(l+1)] +
                                                                                      self.l2_lambd * model.params["W" + str(l+1)]* (1/model.m)  )
            model.params["b" + str(l+1)] = model.params["b" + str(l+1)] - self.lr * (model.grads['db' + str(l+1)] )

class nadam:
    def __init__(self, params, learning_rate = 0.001, beta1= 0.9 ,beta2 = 0.999, l2_lambd=0, epsilon = 1e-8):
        self.lr=learning_rate
        self.beta1=beta1
        self.beta2=beta2
        self.l2_lambd=l2_lambd
        self.epsilon = epsilon
        self.L = len(params) // 2 # number of layers in the neural networks
        self.v = {}
        self.s= {}
        self.t=0;

        for l in range(self.L):
            self.v["dW" + str(l+1)] = np.zeros_like(params['W' + str(l + 1)])
            self.v["db" + str(l+1)] = np.zeros_like(params['b' + str(l + 1)])
            self.s["dW" + str(l+1)] = np.zeros_like(params["W" + str(l + 1)])
            self.s["db" + str(l+1)] = np.zeros_like(params["b" + str(l + 1)])

        self.v_corrected = {}
        self.s_corrected = {}
        
    def update_params_prior_grad(self, model):
        for l in range(self.L):
                                                                                                                          
            self.t = self.t + 1
            
            self.v["dW" + str(l+1)] = self.beta1 * self.v["dW" + str(l+1)] 
            self.v["db" + str(l+1)] = self.beta1 * self.v["db" + str(l+1)]

            self.v_corrected["dW" + str(l + 1)] = self.v["dW" + str(l + 1)] / (1 - self.beta1**(self.t))
            self.v_corrected["db" + str(l + 1)] = self.v["db" + str(l + 1)] / (1 - self.beta1**(self.t))

            model.params["W" + str(l + 1)] = model.params["W" + str(l + 1)] - self.lr * self.v_corrected["dW" + str(l + 1)] 
            model.params["b" + str(l + 1)] = model.params["b" + str(l + 1)] - self.lr * self.v_corrected["db" + str(l + 1)]
            
    def update_params_after_grad(self, model):
        for l in range(self.L):

            self.v["dW" + str(l+1)] = self.v["dW" + str(l+1)] + (1 - self.beta1) * (model.grads["dW" + str(l+1)] +
                                                                                     self.l2_lambd * model.params["W" + str(l+1)]* (1/model.m)  )
            self.v["db" + str(l+1)] = self.v["db" + str(l+1)] + (1 - self.beta1) * (model.grads['db' + str(l + 1)] )

            self.v_corrected["dW" + str(l + 1)] = (1 - self.beta1) * (model.grads["dW" + str(l+1)] +
                                                                       self.l2_lambd * model.params["W" + str(l+1)]* (1/model.m)  ) / (1 - self.beta1**(self.t))
            self.v_corrected["db" + str(l + 1)] = (1 - self.beta1) * model.grads['db' + str(l + 1)] / (1 - self.beta1**(self.t))

            self.s["dW" + str(l + 1)] = self.beta2 * self.s["dW" + str(l + 1)] + (1 - self.beta2) * (model.grads["dW" + str(l+1)]  +
                                                                                                     self.l2_lambd * model.params["W" + str(l+1)]* (1/model.m)  )**2
            self.s["db" + str(l + 1)] = self.beta2 * self.s["db" + str(l + 1)] + (1 - self.beta2) * (model.grads['db' + str(l + 1)] )**2
            
            
            self.s_corrected["dW" + str(l + 1)] = self.s["dW" + str(l + 1)] / (1 - self.beta2**(self.t))
            self.s_corrected["db" + str(l + 1)] = self.s["db" + str(l + 1)] / (1 - self.beta2**(self.t))
            
            
            model.params["W" + str(l + 1)] = model.params["W" + str(l + 1)] - self.lr * self.v_corrected["dW" + str(l + 1)] / (np.sqrt(self.s_corrected["dW" + str(l + 1)]) +
                                                                                                                               self.epsilon)
            model.params["b" + str(l + 1)] = model.params["b" + str(l + 1)] - self.lr * self.v_corrected["db" + str(l + 1)] / (np.sqrt(self.s_corrected["db" + str(l + 1)]) +
                                                                                                                               self.epsilon)
