import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import sys
from data_loader import load_and_prepare_cifar10
from model import NeuralNetwork

log_path=os.path.join(os.path.dirname(__file__),'output.txt')
log_file=open(log_path,'w',encoding='utf-8',buffering=1)
sys.stdout=log_file
sys.stderr=log_file
def save_model(model,path='best_model.pkl'):
    with open(path,'wb') as f:
        pickle.dump(model.get_params(),f)
def load_model(model,path='best_model.pkl'):
    with open(path,'rb') as f:
        params=pickle.load(f)
        model.set_params(params)
def compute_accuracy(preds,labels):
    return np.mean(preds==labels)
def train(model,X_train,y_train,X_val,y_val,num_epochs=20,batch_size=128,learning_rate=1e-2,lr_decay=0.95,print_every=1,save_path='best_model.pkl'):
    num_train=X_train.shape[0]
    iterations_per_epoch=max(num_train//batch_size,1)
    best_val_acc=0.0
    history={
        'train_loss':[],
        'val_loss':[],
        'val_acc':[]
    }
    for epoch in range(1,num_epochs+1):
        indices=np.random.permutation(num_train)
        X_train_shuffled=X_train[indices]
        y_train_shuffled=y_train[indices]
        epoch_loss=0.0
        for i in range(iterations_per_epoch):
            start=i*batch_size
            end=(i+1)*batch_size
            X_batch=X_train_shuffled[start:end]
            y_batch=y_train_shuffled[start:end]
            loss,grads=model.compute_loss_and_gradients(X_batch,y_batch)
            epoch_loss+=loss
            for key in model.params:
                model.params[key]-=learning_rate*grads[key]
        learning_rate*=lr_decay
        avg_train_loss=epoch_loss/iterations_per_epoch
        val_preds=model.predict(X_val)
        val_acc=compute_accuracy(val_preds,y_val)
        probs_val,_=model.forward(X_val)
        val_loss=model._cross_entropy_loss(probs_val,y_val)
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        if print_every and epoch %print_every==0:
            print(f"[epoch {epoch}] train loss:{avg_train_loss:.4f},val loss:{val_loss:.4f},val acc:{val_acc:.4f}")
        if val_acc>best_val_acc:
            best_val_acc=val_acc
            save_model(model,save_path)
            print(f"save model at epoch {epoch} with val acc:{val_acc:.4f}")
    return history

def grid_search(model_class,X_train,y_train,X_val,y_val,learning_rates,hidden_sizes_options,regs,num_epochs=10,batch_size=128):
    best_model=None
    best_val_acc=0
    best_params={}
    for lr in learning_rates:
        for hidden_sizes in hidden_sizes_options:
            for reg in regs:
                model=model_class(
                    input_size=3072,
                    hidden_sizes=hidden_sizes,
                    output_size=10,
                    activation='relu',
                    weight_scale=5e-2,
                    reg=reg


                )
                print(f"train with lr={lr},hidden_sizes={hidden_sizes},reg={reg}")
                history=train(model,X_train,y_train,X_val,y_val,num_epochs=num_epochs,batch_size=batch_size,learning_rate=lr,print_every=1)
                val_acc=history['val_acc'][-1]
                if val_acc>best_val_acc:
                    best_val_acc=val_acc 
                    best_model=model
                    best_params={'learning_rate':lr,'hidden_sizes':hidden_sizes,'reg':reg}
                    print(f"new best model with validation accuracy:{val_acc:.4f} found")
    print(f"best validation accuracy:{best_val_acc:.4f}")
    print(f"best hyperparameters:{best_params}")
    return best_model,best_params

def plot_history(history):
    epochs=range(1,len(history['train_loss'])+1)
    plt.subplot(1,2,1)
    plt.plot(epochs,history['train_loss'],label='train loss')
    plt.plot(epochs,history['val_loss'],label='val loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('training and validation loss')
    plt.legend()

    plt.subplot(1,2,2)
    plt.plot(epochs,history['val_acc'],label='val accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('validation accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()
def visualize_weights(model):
    W1=model.params.get('W1')
    W1=W1.T
    num_neurons=W1.shape[0]
    plt.figure(figsize=(12,12))
    for j in range(min(num_neurons,64)):
        plt.subplot(8,8,j+1)
        w=W1[j]
        w_norm=(w-np.min(w))/(np.max(w)-np.min(w)+1e-8)
        plt.imshow(w_norm.reshape(32,32,3))
        plt.axis('off')
    plt.tight_layout()
    plt.show()
    

if __name__=='__main__':
    X_train,y_train,X_val,y_val,X_test,y_test=load_and_prepare_cifar10('C:/cifar-10-batches-py')
    model=NeuralNetwork(
        input_size=3072,
        hidden_sizes=[2048,1024],
        output_size=10,
        activation='relu',
        weight_scale=5e-2,
        reg=1e-3
        )
    learning_rates=[1e-2,1e-3,1e-4]
    hidden_sizes_options=[[2048,1024],[1024,512],[512,256]]
    regs=[0.0,1e-3,1e-4]
    best_model,best_params=grid_search(NeuralNetwork,X_train,y_train,X_val,y_val,learning_rates,hidden_sizes_options,regs)
    history=train(best_model,X_train,y_train,X_val,y_val,num_epochs=50,batch_size=128,learning_rate=best_params['learning_rate'],print_every=1)
    plot_history(history)
    visualize_weights(best_model)
sys.stdout.close()
sys.stderr.close() 








