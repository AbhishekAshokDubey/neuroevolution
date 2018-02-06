# -*- coding: utf-8 -*-

import numpy as np
import random
from neuralnets import NeuralNet

class Evolve():
    def __init__(self, max_population = 10, max_generations = 10, possible_features = {}):
        self.max_population = max_population
        self.max_generations = max_generations
        self.possible_features = possible_features
        self.population = []
        
        # setting default allowed features in case not provided
        if not(len(self.possible_features)):
            self.possible_features["layers"] = [2,4,5]  # possible_features["layers"] = 5
            self.possible_features["activation_fns"] = ["","relu","sigm"]
            self.possible_features["neurons"] = [10, 50, 100]  # possible_features["neurons"] = 100

        # in case of single interger, it is treated as max value and everything is included
        if (type(self.possible_features["layers"]) != list):
            self.possible_features["layers"] = list(range(self.possible_features["layers"]))
        if (type(self.possible_features["neurons"]) != list):
            self.possible_features["neurons"] = list(range(self.possible_features["neurons"]))

    def create_population(self, input_dim, output_dim, output_fn, max_population = None):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.output_fn = output_fn
        self.population = []

        if not max_population:
            max_population = self.max_population

        for _ in range(max_population):
            layer_count = np.random.choice(self.possible_features["layers"])
            arch = np.random.choice(self.possible_features["neurons"], size=layer_count, replace=True)
            
            arch = np.append(np.insert(arch,0,input_dim),output_dim)

            activation_fn = np.random.choice(self.possible_features["activation_fns"], size=layer_count, replace=True)
            nn = NeuralNet(arch, activation_fn)
            self.population.append(nn)
    
    def breed(self, parent_nn_1, parent_nn_2):
        parent_1_len = len(parent_nn_1.arch)
        parent_2_len = len(parent_nn_2.arch)
        diff = parent_1_len - parent_2_len
        
        arch = np.random.choice([parent_nn_1.arch, parent_nn_2.arch])
        activation_fn = np.random.choice([parent_nn_1.activation_fn, parent_nn_2.activation_fn])
        
        # in case of dim miss-match we keep arch and change activation fn accordingly
        if len(arch)!=len(activation_fn):
            if len(arch) > len(activation_fn): # Case1: increase the act_fn by borrowing from the other larger one
                if diff > 0:
                    activation_fn += parent_nn_1.activation_fn[-1*diff:]
                else:
                    activation_fn += parent_nn_2.activation_fn[diff:]
            elif len(activation_fn) > len(arch): # Case2: remove the extra values in the act_fn
                activation_fn = activation_fn[:len(arch)]
        
        return NeuralNet(arch, activation_fn)
    
    def mutate(self, nn):
        arch = nn.arch
        activation_fn = nn.activation_fn
        
        mutate_key = random.choice(list(self.possible_features.keys()))
        layer_to_mutate = np.random.choice(range(len(arch)))
        if mutate_key == "layers": # mutate architecture/ layers count
            if np.random.rand() > 0.5: # add a layer
                arch.insert(layer_to_mutate,
                            np.random.choice(self.possible_features["neurons"]))
            else: # remove a layer
                arch.pop(layer_to_mutate)
        if mutate_key == "activation_fns":
            activation_fn[layer_to_mutate] = np.random.choice(self.possible_features["activation_fns"])
        if mutate_key == "neurons":
            arch[layer_to_mutate] = np.random.choice(self.possible_features["neurons"])
        
        return NeuralNet(arch, activation_fn)
    
    # fitness: using loss, but better would be to use rmse & accuracy respectively
    def loss_score(self,nn, x,y):
        y_hat = nn.forward(x)
        return nn.loss(y,y_hat)
    
    def sort_population(self, x, y):
        sorted_population = sorted(self.population, key=lambda individual: self.loss_score(individual,x,y))
        self.population = sorted_population
    
    def train_population(self, x, y, epochs=1):
        for i in range(len(self.population)):
            for _ in range(epochs):
                self.population[i].train(x,y)

    def show_population(self, count=10):
        for indx in range(count):
            self.population[indx].draw();
            print("-----------") 