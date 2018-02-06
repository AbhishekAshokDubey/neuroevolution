# -*- coding: utf-8 -*-

from evolve import Evolve
from neuralnets import generate_random_data


if __name__ == "__main__":
    possible_features = {}
    possible_features["layers"] = [2,4,5]
    possible_features["activation_fns"] = ["","relu","sigm"]
    possible_features["neurons"] = [10, 50, 100]

    x_train, y_train, x_validate, y_validate = generate_random_data()

    smart_nets = Evolve(max_population = 10, max_generations = 10, possible_features = possible_features)
    smart_nets.create_population(x_train.shape[1], y_train.shape[1], "") #smart_nets.population[0].arch
    smart_nets.show_population(3)
    print("----- Lets train and evolve -----")
    smart_nets.train_population(x_train, y_train, epochs=10)
    smart_nets.sort_population(x_validate, y_validate)
    smart_nets.show_population(3)
