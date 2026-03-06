import numpy as np

LEARNING_RATE = 0.001

def create_dataset(n: int):
    dataset: list = []

    for i in range(0, n):
        dataset.append((i, i*i))

    return dataset

def initialize_weights(x,y):
    return np.random.uniform(x,y,1)

def forward(model:list, input:int):
    result = input
    for layer in model:
        result = np.dot(layer, result)
        print(result)
    return result
def calculate_loss(model, dataset):
    MSE = 0
    for (x,y) in dataset:
        MSE += (y - forward(model, x))**2
    return MSE    

def train_model(model, dataset, n_epochs, use_learing_rate = False, print_all = False):
    for i in range(1, n_epochs+1):

        MSE = calculate_loss(model, dataset)

        if print_all:
            print("Epoch:", i)
            print("MSE:", MSE)

        for i, layer in enumerate(model):
            for j, neuron in enumerate(model):
                altered_model = np.copy(model)
                altered_model[i][j] += 0.001
                plus_eps_MSE = calculate_loss(altered_model, dataset)
                L = plus_eps_MSE - MSE
                neuron -= (L*LEARNING_RATE) if use_learing_rate else L
    print("Final MSE:", MSE)  

def main():
    dataset = create_dataset(100)

    model_shape = (2,2)
    model = []
    for i in range(model_shape[0]):
        layer = []
        for j in range(model_shape[1]):
            layer.append(initialize_weights(1,100))
        model.append(layer)
    
    model = np.array(model)
    
    train_model(model,dataset,10000)

if __name__ == "__main__":
    main()
