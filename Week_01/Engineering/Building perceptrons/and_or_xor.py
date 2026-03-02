import numpy as np

LEARNING_RATE = 0.001

def create_dataset(func):
    dataset: list = []

    for i in range(2):
        for j in range(2):
            dataset.append((i, j, func(i,j) ))
    print("Dataset:", dataset)

    return dataset

def initialize_weights(x,y):
    return np.random.uniform(x,y,1)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def calculate_loss(model, dataset):
    MSE = 0
    for (x1, x2, y) in dataset:
        activation = sigmoid(x1*model[0] + x2*model[1] + model[2])
        prediction = round(activation)
        
        MSE += (y - activation)**2
    return MSE    

def train_model(model, dataset, n_epochs, use_learing_rate = False, print_all = False):
    for i in range(1, n_epochs+1):
        MSE = calculate_loss(model, dataset)

        if print_all:
            print("Epoch:", i)
            print("MSE:", MSE)

        for i in range(3):
            altered_model = np.copy(model)
            altered_model[i] += 0.001
            plus_eps_MSE = calculate_loss(altered_model, dataset)
            L = plus_eps_MSE - MSE
            model[i] -= (L*LEARNING_RATE) if use_learing_rate else L
    print("Final MSE:", MSE)  
    for (i, param) in enumerate(model):
        print(f"Parameter {i}: {param}")

#General form of OR and AND neural networks:
#There are 2 inputs X1 and X2 and each one has a weight. Also there is one bias. We use the sigmoid function to contain the output values in the range [0,1]

def main():
    or_dataset = create_dataset(lambda x,y: x or y)
    or_model = []
    for i in range(3):
        or_model.append(initialize_weights(-1,1))
    
    print("OR neural network!")
    print("Before training:")
    for (i, param) in enumerate(or_model):
        print(f"Parameter {i}: {param}")
    train_model(or_model, or_dataset, 10000)
    print("After training:")
    for (i, param) in enumerate(or_model):
        print(f"Parameter {i}: {param}")

    and_dataset = create_dataset(lambda x,y: x and y)
    and_model = []
    for i in range(3):
        and_model.append(initialize_weights(-1,1))
    print("AND neural network!")
    print("Before training:")
    for (i, param) in enumerate(and_model):
        print(f"Parameter {i}: {param}")
    train_model(and_model, and_dataset, 10000)
    print("After training:")
    for (i, param) in enumerate(and_model):
        print(f"Parameter {i}: {param}")

if __name__ == "__main__":
    main()