import numpy as np
import matplotlib.pyplot as plt
from activation_fns_and_grads import *
from loss_fns_with_regularization import *
from optimizers_with_regularization import *
from utils import *


# Prior to this define the model
def train(model,
          X_train,
          Y_train,
          X_val,
          Y_val,
          output_activation="sigmoid", 
          output_loss_fn="cross_entropy", 
          epochs=1, 
          optimizer_name='adam',
          learning_rate=0.0075, 
          l2_lambd = 0,
          lr_schedule = 1,
          batch_size=2, 
          print_cost=False, 
          val=True):
    
    train_costs_batches = []
    train_accuracy_batches = []
    
    train_costs=[]
    train_accuracy = []
    val_costs = []
    val_accuracy = []
    
    #X_train, X_val, Y_train, Y_val = train_val_split(X, Y, val_size=0.1, random_state=42)
    
    if optimizer_name=='sgd':
        optimizer = stochastic_gradient_descent(learning_rate=learning_rate , l2_lambd=l2_lambd)
    elif optimizer_name=='momentum':
        optimizer = momentum_gradient_descent(params=model.params,learning_rate= learning_rate , l2_lambd=l2_lambd,gamma=0.9)
    elif optimizer_name=='rmsprop':
        optimizer = rms_prop(params=model.params,learning_rate= learning_rate , l2_lambd=l2_lambd , beta2 = 0.999)
    elif optimizer_name=='adam':
        optimizer = adam(params = model.params, learning_rate= learning_rate , l2_lambd=l2_lambd, beta1 = 0.9, beta2=0.999)
    elif optimizer_name== 'nesterov':
        optimizer = nesterov_gradient_descent(params = model.params, learning_rate= learning_rate , l2_lambd=l2_lambd , gamma=0.9)
    elif optimizer_name=='nadam':
        optimizer = nadam(params = model.params, learning_rate= learning_rate, beta1=0.9, beta2=0.999 , l2_lambd=l2_lambd)
        
    if batch_size==None:
        batch_size = X_train.shape[1]
        
    for i in range(1,epochs+1):
        mini_batches = create_mini_batches(X_train, Y_train, batch_size)
        no_batches = len(mini_batches)
        
        epoch_loss=0
        epoch_correct=0

        if i%10==0:
            optimizer.lr=optimizer.lr * lr_schedule
        '''decay = 0.001
        optimizer.lr=optimizer.lr * 1/(1 + decay * i)'''
        for j , mini_batch in enumerate(mini_batches,1):
            X_mini, Y_mini = mini_batch
            bs = X_mini.shape[1]
            if bs==0:
                break
            
            # Forward Propagation
            XL = model.forward_prop(X_mini) ## final layer output w/o activation
            Y_pred = act_fn(XL,output_activation)
            
            # Compute cost
            batch_cost = cost_fn(model.params,Y_mini,Y_pred, l2_lambd=l2_lambd, output_activation=output_activation, loss_fn=output_loss_fn)
            batch_correct = np.sum(onehot_to_class(Y_pred)==Y_mini)
            
            epoch_loss += batch_cost*bs
            epoch_correct += batch_correct
            
            # Backward propagation
            ## look forward step
            if optimizer_name=='nesterov' or optimizer_name=='nadam':
                optimizer.update_params_prior_grad(model)
            
            dZ=output_exc_act_grad(Y_mini,Y_pred,output_activation=output_activation,loss_fn=output_loss_fn,classes=model.output_nodes)

            model.backward_prop(dZ)

            # Update parameters.
            if optimizer_name== 'nesterov' or optimizer_name=='nadam':
                optimizer.update_params_after_grad(model)
            else:
                optimizer.update_params(model)

            # Print the cost every 100 training batch
            
            if j % 100 == 0:
                #print("Cost after iteration {}: {}".format(i, np.squeeze(batch_cost)))
                train_costs_batches.append(batch_cost)
                train_accuracy_batches.append(batch_correct/Y_mini.shape[1])
                #wandb.log({'batch_no':i*j, 'train loss per 100 batch': batch_cost})
                #wandb.log({'epoch': i,'train loss': batch_cost})
        
        epoch_loss = epoch_loss / X_train.shape[1]
        epoch_correct = epoch_correct / X_train.shape[1]
        
        train_costs.append(epoch_loss)
        train_accuracy.append(epoch_correct)

        if val :
            val_acc,val_loss=accuracy_loss(model,X_val,Y_val,output_activation=output_activation,output_loss_fn=output_loss_fn)
            val_costs.append(val_loss)
            val_accuracy.append(val_acc)
            #print("batch_cost",batch_cost) # last batch cost of epoch
            
            if print_cost:
                print("val cost for epoch ", i , "= ",val_loss)
                print("val accuracy for epoch ", i , "= ",val_acc)
                print("train cost for epoch ", i , "= ",epoch_loss)
                print("train accuracy for epoch ", i , "= ",epoch_correct)
                print("")
            
    # plot the cost per 100 batches
    plt.plot(np.squeeze(train_costs_batches))
    plt.plot(np.squeeze(train_accuracy_batches))
    plt.ylabel('train_costs_batches')
    plt.xlabel('batches (per 100)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
