# -*- coding: utf-8 -*-

# if you new to this, few good to know
# https://github.com/HIPS/autograd
# http://ruder.io/optimizing-gradient-descent/
# https://github.com/AbhishekAshokDubey/RL/tree/master/ping-pong

import numpy as np

class NeuralNet():
    def __init__(self, arch = [10,5,1], activation_fn = ['', 'relu', 'sigm'],
                 loss_fn = "sq_loss", learning_rate = 1e-4, decay_rate = 0.99, eps = 1e-5):
        self._weights = []
        self.rmsprop_mem = []
        self.arch = arch  
        self.loss_fn = loss_fn
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.eps = eps

        self.layer_inputs = []
        self.layer_outputs = []

        if(len(arch)!=len(activation_fn)):
            self.activation_fn = len(arch)*['']
        else:
            self.activation_fn = activation_fn

        for l_i, l_ip1 in zip(arch[:-1], arch[1:]):
            self._weights.append(np.random.rand(l_ip1, l_i))
            self.rmsprop_mem.append(np.zeros(shape=(l_ip1, l_i)))
    
    def activation(self, x, fn, forward = True):
        if(fn == "sigm"):
            if forward:
                return 1 / (1 + np.exp(-x))
            else:
                return x * (1 - x)
        elif(fn == "relu"):
            if forward:
                return np.maximum(x, 0) # x * (x > 0)
            else:
                return 1. * (x > 0)
        else: #assumes no activation
            if forward:
                return x
            else:
                return np.ones_like(x)

    def loss(self, true_output, pred_output=None):
        if pred_output is None:
            pred_output = self.layer_outputs[-1]
        if(self.loss_fn == "sq_loss"): # regression
            return 0.5 * np.mean(np.sum(np.square(pred_output - true_output), axis=1))
        elif(self.loss_fn == "log_loss"): # multi label classification error, not exactly binary nor softmax
            return -1. * np.mean(np.sum(true_output * np.log(pred_output), axis=1))
    
    def forward(self,input_batch):
        self.layer_inputs = []
        self.layer_outputs = []

        self.layer_inputs.append(input_batch)
        self.layer_outputs.append(self.activation(input_batch, self.activation_fn[0]))

        for i, w in enumerate(self._weights):
            self.layer_inputs.append(np.dot(self.layer_inputs[-1],w.T))
            self.layer_outputs.append(self.activation(self.layer_inputs[-1], self.activation_fn[i]))
        
        return self.layer_outputs[-1]

    def backward(self, true_output):
        delta_weights = np.zeros_like(self._weights)
        if(self.loss_fn == "sq_loss"):
            loss = -1. * np.sum(self.layer_outputs[-1] - true_output, axis=1)
        elif(self.loss_fn == "log_loss"):
            loss = -1. (true_output/ self.loss_fn[-1])

        if len(loss.shape) == 1:
            loss = loss.reshape(-1,1)
        
        for indx in reversed(range(len(self._weights))):
            loss = loss * self.activation(self.layer_outputs[indx+1], self.loss_fn[indx+1], forward = False)
            delta_weights[indx] = np.dot(loss.T, self.layer_outputs[indx])
            loss = np.dot(loss, self._weights[indx])
        return delta_weights

    def update(self, grad_list):
        for i, dw in enumerate(grad_list):
            self.rmsprop_mem[i] = self.decay_rate * self.rmsprop_mem[i] + ((1-self.decay_rate) * dw**2)
            self._weights[i] += self.learning_rate * dw / (np.sqrt(self.rmsprop_mem[i]) + self.eps)
    
    def train(self, x,y):
        self.forward(x)
        delta_weights = self.backward(y)
        self.update(delta_weights)
        return self.loss(y)
        
    def draw(self):
        layer_count = len(self.arch)
        for i, layer in enumerate(reversed(list(zip(self.arch, self.activation_fn)))):
            if i==0:
                print("o/p",layer[0],layer[1])
            elif i == layer_count-1:
                print("i/p",layer[0],layer[1])
            else:
                print(" hd",layer[0],layer[1])

##################################################################
# just a helper function
def generate_random_data(split=0.8):
    x = np.random.rand(1200,5)
    y = np.sum(x, axis=1)
    y = y.reshape(-1,1)

    # indexes = np.random.permutation(x.shape[0])
    split_indx = int(split * x.shape[0])

    x_train, x_validate = x[:split_indx,:], x[split_indx:,:]
    y_train, y_validate = y[:split_indx,:], y[split_indx:,:]
    
    return (x_train, y_train, x_validate, y_validate)