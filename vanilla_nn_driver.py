# -*- coding: utf-8 -*-

import matplotlib.pylab as plt
from neuralnets import NeuralNet, generate_random_data
import numpy as np


def plot_training(train_loss, val_loss, skip = 50):
    plt.plot(np.asmatrix(range(len(train_loss))).T * skip, train_loss, '-g', lw=2, label='train-loss')
    plt.plot(np.asmatrix(range(len(val_loss))).T * skip, val_loss, '-r', lw=2, label='val-loss')

if __name__ == "__main__":
    epochs = 1000
    nn = NeuralNet(arch = [5,2,1], activation_fn = ['', '', ''],learning_rate = 1e-4)
    x_train, y_train, x_validate, y_validate = generate_random_data()
    train_loss = []
    val_loss = []
    plot_skip = 50
    print("\n training begins ...\n")
    for i in range(epochs):
        loss = nn.train(x_train,y_train)
        if not(i%plot_skip):
            train_loss.append(loss)
            y_validate_hat = nn.forward(x_validate)
            loss_val = nn.loss(y_validate)
            val_loss.append(loss_val)
#            print("-----------")
    print("\n training ends ...\n")
    print("train loss:", train_loss[-1])
    print("val loss:", val_loss[-1])
    plot_training(train_loss, val_loss, plot_skip)