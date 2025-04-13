import numpy as np
class NeuralNetwork:
    def __init__(self,input_size,hidden_sizes,output_size,activation='relu',weight_scale=1e-3,reg=0.0):
        self.hidden_sizes=hidden_sizes
        self.activation_type=activation
        self.reg=reg
        self.params={}
        layer_dims=[input_size]+hidden_sizes+[output_size]
        for i in range(len(layer_dims)-1):
            self.params[f'W{i+1}']=np.random.randn(layer_dims[i],layer_dims[i+1])*weight_scale
            self.params[f'b{i+1}']=np.zeros((1,layer_dims[i+1]))
    def _activation(self,x):
        if self.activation_type=='relu':
            return np.maximum(0,x)
        elif self.activation_type=='sigmoid':
            return 1/(1+np.exp(-x))
        else:
            raise ValueError("请选择relu或者sigmoid激活函数")
    def _activation_derivative(self,x):
        if self.activation_type=='relu':
            return (x>0).astype(float)
        elif self.activation_type=='sigmoid':
            sig=1/(1+np.exp(-x))
            return sig*(1-sig)
        else:
            raise ValueError("请选择relu或者sigmoid激活函数")
    def _softmax(self,x):
        x_shifted=x-np.max(x,axis=1,keepdims=True)
        exp_scores=np.exp(x_shifted)
        return exp_scores/np.sum(exp_scores,axis=1,keepdims=True)
    def _cross_entropy_loss(self,probs,y):
        N=y.shape[0]
        correct_logprobs=-np.log(probs[np.arange(N),y]+1e-8)
        data_loss=np.sum(correct_logprobs)/N
        return data_loss
    def forward(self,X):
        cache={}
        A=X
        cache['A0']=A
        num_layers=len(self.hidden_sizes)+1
        for i in range(1,num_layers):
            Z=A@self.params[f'W{i}']+self.params[f'b{i}']
            A=self._activation(Z)
            cache[f'Z{i}']=Z
            cache[f'A{i}']=A
        Z_final=A@self.params[f'W{num_layers}']+self.params[f'b{num_layers}']
        probs=self._softmax(Z_final)
        cache[f'Z{num_layers}']=Z_final
        cache[f'A{num_layers}']=probs
        return probs,cache
    def compute_loss_and_gradients(self,X,y):
        grads={}
        N=X.shape[0]
        num_layers=len(self.hidden_sizes)+1
        probs,cache=self.forward(X)
        data_loss=self._cross_entropy_loss(probs,y)
        reg_loss=0.0
        for i in range(1,num_layers+1):
            W=self.params[f'W{i}']
            reg_loss+=0.5*self.reg*np.sum(W*W)
        loss=data_loss+reg_loss
        dZ=probs
        dZ[np.arange(N),y]-=1
        dZ/=N
        for i in reversed(range(1,num_layers+1)):
            A_prev=cache[f'A{i-1}']
            grads[f'W{i}']=A_prev.T@dZ+self.reg*self.params[f'W{i}']
            grads[f'b{i}']=np.sum(dZ,axis=0,keepdims=True)
            if i>1:
                dA_prev=dZ@self.params[f'W{i}'].T
                Z_prev=cache[f'Z{i-1}']
                dZ=dA_prev*self._activation_derivative(Z_prev)
        return loss,grads
    def predict(self,X):
        probs,_=self.forward(X)
        return np.argmax(probs,axis=1)
    def get_params(self):
        return self.params
    def set_params(self,params):
        self.params=params
        

