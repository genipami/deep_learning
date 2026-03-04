import numpy as np
import matplotlib.pyplot as plt

EPS = 0.001
LEARNING_RATE = 0.001

def create_dataset(func) -> list:
    dataset: list = []

    for i in range(2):
        for j in range(2):
            dataset.append((i, j, func(i,j) ))
    print("Dataset:", dataset)

    return dataset

def initialize_weights(x:float, y:float) -> float:
    return np.random.uniform(x,y)

def sigmoid(x:float) -> float:
    return 1 / (1 + np.exp(-x))

def calculate_loss(model: list, dataset: list) -> float:
    MSE = 0
    for (x1, x2, y) in dataset:
        activation = sigmoid(x1*model[0] + x2*model[1] + model[2])
        
        MSE += (y - activation)**2
    return MSE    

def train_model(model:list, dataset:list, n_epochs:int, use_learing_rate: bool = False, print_all: bool = False) -> list:
    losses = []
    for i in range(1, n_epochs+1):
        MSE = calculate_loss(model, dataset)
        losses.append(MSE)

        if print_all:
            print("Epoch:", i)
            print("MSE:", MSE)

        for i in range(3):
            altered_model = np.copy(model)
            altered_model[i] += EPS
            plus_eps_MSE = calculate_loss(altered_model, dataset)
            L = (plus_eps_MSE - MSE)/EPS
            model[i] -= (L*LEARNING_RATE) if use_learing_rate else L
    print("Final MSE:", MSE)  
    for (i, param) in enumerate(model):
        print(f"Parameter {i}: {param}")
    
    return losses

#General form of OR and AND neural networks:
#There are 2 inputs X1 and X2 and each one has a weight. Also there is one bias. We use the sigmoid function to contain the output values in the range [0,1]
def print_results(model: list, dataset: list) -> None:
    for (x1, x2, y) in dataset:
        print("Expected output:", y, "Prediction:", sigmoid(x1*model[0] + x2*model[1] + model[2]))

def create_single_layer_bool_nn(func):
    dataset = create_dataset(func)
    model = []

    for i in range(3):
        model.append(initialize_weights(-1,1))

    print("Before training:")
    for (i, param) in enumerate(model):
        print(f"Parameter {i}: {param}")

    and_losses = train_model(model, dataset, 10000, True)
    plt.plot(and_losses)
    plt.title("Change in MSE over 10 000 epochs")
    plt.xlabel("Epochs")
    plt.ylabel("MSE")
    plt.show()

    print("After training:")
    for (i, param) in enumerate(model):
        print(f"Parameter {i}: {param}")
    
    return model

def main():
    
    or_model = create_single_layer_bool_nn(lambda x,y : x or y)
    or_dataset = create_dataset(lambda x,y : x or y)

    print("Results from OR network:")
    print_results(or_model, or_dataset)

    and_model = create_single_layer_bool_nn(lambda x,y : x and y)
    and_dataset = create_dataset(lambda x,y : x and y)

    print("Results from AND network:")
    print_results(and_model, and_dataset)

    nand_model = create_single_layer_bool_nn(lambda x,y : not(x and y))
    nand_dataset = create_dataset(lambda x,y : not(x and y))
    print("Results from NAND network:")
    print_results(nand_model, nand_dataset)

    

if __name__ == "__main__":
    main()