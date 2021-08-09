from network import model
import numpy as np

def read_model():
    data = open('./models/model1/data.txt', 'r')
    num_inputs = int(data.readline())
    num_intermediate_layers = int(data.readline())
    layer_height = int(data.readline())
    num_outputs = int(data.readline())
    mutation_chace = int(data.readline())
    weights = []
    biases = []
    for i in range(num_intermediate_layers + 1):
        weights.append(np.genfromtxt(f'./models/model1/layer{i}/weights.txt', delimiter=','))
        biases.append(np.genfromtxt(f'./models/model1/layer{i}/biases.txt', delimiter=','))
    return model(weights=weights, biases=biases, mutation_chance=mutation_chace)

if __name__ == '__main__':
    my_model = read_model()
    input = np.array([0.4, 0.4, 0.4, 0.4, 0.0])
    output = my_model.get_output(input)
    print(output)

