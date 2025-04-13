import numpy as np
import pickle
from data_loader import load_and_prepare_cifar10
from model import NeuralNetwork

def compute_accuracy(preds,labels):
    return np.mean(preds==labels)
def load_model(model,path='best_model.pkl'):
    with open(path,'rb') as f:
        params=pickle.load(f)
        model.set_params(params)
def test(model,X_test,y_test):
    preds=model.predict(X_test)
    test_acc=compute_accuracy(preds,y_test)
    print(f"Test Accuracy:{test_acc:.4f}")
    return test_acc
if __name__=='__main__':
    X_train,y_train,X_val,y_val,X_test,y_test=load_and_prepare_cifar10('C:/cifar-10-batches-py')
    model=NeuralNetwork(
        input_size=3072,
        hidden_sizes=[2048,1024],
        output_size=10,
        activation='relu',
        weight_scale=5e-2,
        reg=0.0
    )
    load_model(model,'best_model.pkl')
    test_acc=test(model,X_test,y_test)