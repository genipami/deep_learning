import numpy as np

LEARNING_RATE = 0.001
EPS = 0.001

def create_dataset(n: int) -> list:
    dataset: list = []

    for i in range(0, n):
        dataset.append((i, i*i))

    return dataset

def initialize_weights(x,y) -> float:
    return np.random.uniform(x,y)

def forward(model, x):
    weights, biases = model
    
    result = np.array([x], dtype=float)

    for weight, bias in zip(weights, biases):
        result = weight @ result + bias
        result = np.maximum(0, result)

    return result[0]
def calculate_loss(model, dataset):
    MSE = 0
    for (x,y) in dataset:
        MSE += (y - forward(model, x))**2
    MSE /= len(dataset)
    return MSE  

def train_model(model, dataset, n_epochs, use_learing_rate = False, print_all = False):
    print("MSE before:", calculate_loss(model, dataset))

    for i in range(1, n_epochs+1):

        MSE = calculate_loss(model, dataset)

        if print_all:
            print("Epoch:", i)
            print("MSE:", MSE)

        (weights, biases) = model
        for i, layer in enumerate(weights):
            for j, neuron in enumerate(layer):
                altered_model = np.copy(model)
                altered_model[0][i][j] += EPS
                plus_eps_MSE = calculate_loss(altered_model, dataset)
                L = (plus_eps_MSE - MSE)/EPS
                model[0][i][j] -= (L*LEARNING_RATE) if use_learing_rate else L

        for i, layer in enumerate(biases):
            for j, bias in enumerate(layer):
                altered_model = np.copy(model)
                altered_model[1][i][j] += EPS
                plus_eps_MSE = calculate_loss(altered_model, dataset)
                L = (plus_eps_MSE - MSE)/EPS
                model[1][i][j] -= (L*LEARNING_RATE) if use_learing_rate else L
        
    print("Final MSE:", MSE)  

def main():
    dataset_size = 100
    dataset = create_dataset(dataset_size)

    weights_shape = (2,2)
    weights = []
    for i in range(weights_shape[0]):
        layer = []
        for j in range(weights_shape[1]):
            layer.append(initialize_weights(0,1))
        weights.append(layer)
    
    weights = np.array(weights)

    biases = np.ones(weights_shape)
    model = (weights, biases)
    train_model(model,dataset,10000)

if __name__ == "__main__":
    main()
