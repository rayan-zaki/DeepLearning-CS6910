import numpy as np
from sklearn.model_selection import train_test_split
from activation_fns_and_grads import *
from loss_fns_with_regularization import *

def class_onehot(indices, num_classes , zero_indexed=True): # Takes class number [0,classes-1] and returns one hot (zero indexed)
    #print(indices)
    indices = indices.astype('int') 
    if zero_indexed:
        return np.identity(num_classes)[np.array(indices)].T.squeeze()
    else :
        return np.identity(num_classes)[np.array(indices-1)].T.squeeze()

def onehot_to_class(y_pred):
    return np.argmax(y_pred,axis=0).reshape(1,-1)

def train_val_split(X, Y, val_size=0.1, random_state=42):
    X_train, X_val, Y_train, Y_val =train_test_split(X.T, Y.T, test_size=val_size, random_state=random_state)
    return X_train.T, X_val.T, Y_train.T, Y_val.T

def accuracy_loss(model,X,Y,output_activation,output_loss_fn):
    loss=0
    correct=0
    total=X.shape[1]
    for i in range(X.shape[1]):
        XL=model.forward_prop(X[:,i].reshape(-1,1))
        Y_pred=act_fn(XL,output_activation)
        class_pred=onehot_to_class(Y_pred)[0][0]
        if class_pred==Y[0][i]:
            correct+=1
        
        loss+=cost_fn(model.params,Y[:,i].reshape(-1,1),Y_pred.squeeze(), output_activation=output_activation, loss_fn=output_loss_fn)#, classes=model.output_nodes)
    return correct/total,loss/total

def create_mini_batches(X, Y, batch_size): 
    mini_batches = [] 
    data = np.vstack((X, Y)).T
    np.random.shuffle(data) 
    n_minibatches = data.shape[0] // batch_size 
    i = 0
  
    for i in range(n_minibatches + 1): 
        mini_batch = data[i * batch_size:(i + 1)*batch_size, :] 
        X_mini = mini_batch[:, :-1].T
        Y_mini = mini_batch[:, -1].reshape((-1, 1)).T 
        mini_batches.append((X_mini, Y_mini)) 
    if data.shape[0] % batch_size != 0: 
        mini_batch = data[i * batch_size:data.shape[0]] 
        X_mini = mini_batch[:, :-1].T
        Y_mini = mini_batch[:, -1].reshape((-1, 1)).T 
        mini_batches.append((X_mini, Y_mini)) 
    return mini_batches

def output_exc_act_grad(y,y_pred, output_activation="sigmoid", loss_fn="cross_entropy", classes=1):
    if output_activation=="sigmoid" and loss_fn=="cross_entropy":
        return y_pred-y
    elif output_activation=="softmax" and loss_fn=="cross_entropy":
        y=class_onehot(y,classes)
        return y_pred-y 
    elif output_activation=="linear" and loss_fn=="mse":
        return y - y_pred
