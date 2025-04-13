import os
import pickle
import numpy as np

def load_data_batch(file_path):
    with open(file_path,'rb') as f:
        batch=pickle.load(f,encoding='latin1')
    return batch['data'],batch['labels']
def load_cifar10_data(data_dir):
    train_data=[]
    train_labels=[]
    for i in range(1,6):
        file_path=os.path.join(data_dir,f"data_batch_{i}")
        data,labels=load_data_batch(file_path)
        train_data.append(data)
        train_labels.extend(labels)
    train_data=np.vstack(train_data)
    train_labels=np.array(train_labels)

    test_file=os.path.join(data_dir,"test_batch")
    test_data,test_labels=load_data_batch(test_file)
    test_data=np.array(test_data)
    test_labels=np.array(test_labels)
    return train_data,train_labels,test_data,test_labels
def normalize_data(data):
    data=data.astype(np.float32)/255.0
    mean_image=np.mean(data,axis=0)
    return data-mean_image
def train_val_split(X,y,val_ratio=0.1,seed=42):
    np.random.seed(seed)
    indices=np.arange(X.shape[0])
    np.random.shuffle(indices)
    split=int(X.shape[0]*(1-val_ratio))
    train_idx=indices[:split]
    val_idx=indices[split:]
    return X[train_idx],y[train_idx],X[val_idx],y[val_idx]
def load_and_prepare_cifar10(data_dir,val_ratio=0.1):
    X_train_raw,y_train_raw,X_test,y_test=load_cifar10_data(data_dir)
    X_train_raw=normalize_data(X_train_raw)
    X_test=normalize_data(X_test)
    X_train,y_train,X_val,y_val=train_val_split(X_train_raw,y_train_raw,val_ratio=val_ratio)
    return X_train,y_train,X_val,y_val,X_test,y_test