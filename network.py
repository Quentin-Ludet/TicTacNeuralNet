import numpy as np
import random
import math

def sigmoid(x):
    return 1/(1+(np.e**(-x)))

def solve_layer(input, biases, weights):
    product = np.matmul(weights, input)
    sum = np.add(product, biases)
    output = np.array([sigmoid(x) for x in sum])
    return output

class model:
    num_inputs = 18
    num_intermediate_layers = 3
    layer_height = 12
    num_outputs = 9

    class layer:
        def __init__(self, count, inputs, random=False, weights=None, biases=None):
            if random:
                self.weights = np.random.rand(count, inputs) * 2 - 1
                self.biases = np.random.rand(count) * 2 - 1
            else:
                self.weights = weights
                self.biases = biases


    def __init__(self, random=False, weights=None, biases=None, mutation_chance=None, name=None):
        self.name = name
        self.layers = []
        self.calculated_cost = None
        if random:
            self.layers.append(self.layer(self.layer_height, self.num_inputs, random=True))
            for i in range(self.num_intermediate_layers - 1):
                self.layers.append(self.layer(self.layer_height, self.layer_height, random=True))
            self.layers.append(self.layer(self.num_outputs, self.layer_height, random=True))
        else:
            self.layers.append(self.layer(self.layer_height, self.num_inputs, weights=weights[0], biases=biases[0]))
            for i in range(self.num_intermediate_layers - 1):
                self.layers.append(self.layer(self.layer_height, self.layer_height, weights=weights[i+1], biases=biases[i+1]))
            self.layers.append(self.layer(self.num_outputs, self.layer_height, weights=weights[-1], biases=biases[-1]))
        self.mutation_chance = mutation_chance

    def get_output(self, input):
        layer_sols = [input]
        for i in range(self.num_intermediate_layers + 1):
            layer_sols.append(solve_layer(layer_sols[-1], self.layers[i].biases, self.layers[i].weights))
        return layer_sols[-1]

    def create_child(father, mother):
        new_weights = []
        new_biases = []
        for n in range(len(father.layers)):
            new_weights.append(np.zeros(father.layers[n].weights.shape))
            for i in range(father.layers[n].weights.shape[0]):
                for j in range(father.layers[n].weights.shape[1]):
                    if bool(random.getrandbits(1)):
                        new_weights[n][i][j] = father.layers[n].weights[i][j]
                    else:
                        new_weights[n][i][j] = mother.layers[n].weights[i][j]
                    if random.randrange(0,father.mutation_chance) == 0:
                        new_weights[n][i][j] = 2 * (random.random() * 2 - 1) * father.layers[n].weights[i][j]
            new_biases.append(np.zeros(father.layers[n].biases.shape))
            for i in range(father.layers[n].biases.shape[0]):
                if bool(random.getrandbits(1)):
                    new_biases[n][i] = father.layers[n].biases[i]
                else:
                    new_biases[n][i] = mother.layers[n].biases[i]
                if random.randrange(0,father.mutation_chance) == 0:
                    new_biases[n][i] = 2 * (random.random() * 2 - 1) * father.layers[n].biases[i]
        if bool(random.getrandbits(1)):
            new_mutation_chance = father.mutation_chance
        else:
            new_mutation_chance = mother.mutation_chance
        if random.randrange(0,father.mutation_chance) == 0:
            new_mutation_chance = math.ceil(2 * random.random() * father.mutation_chance)
        return model(weights=new_weights, biases=new_biases, mutation_chance=new_mutation_chance)

    def save(self):
        data = open('./models/model1/data.txt', 'w')
        data.write(str(self.num_inputs) + '\n')
        data.write(str(self.num_intermediate_layers) + '\n')
        data.write(str(self.layer_height) + '\n')
        data.write(str(self.num_outputs) + '\n')
        data.write(str(self.mutation_chance))
        for i in range(self.num_intermediate_layers+1):
            np.savetxt(f'./models/model1/layer{i}/weights.txt', self.layers[i].weights, delimiter=',')
            np.savetxt(f'./models/model1/layer{i}/biases.txt', self.layers[i].biases, delimiter=',')