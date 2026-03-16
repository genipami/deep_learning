import numpy as np

LEARNING_RATE = 0.001
rng = np.random.default_rng(42)
def create_dataset(n: int) -> list:
    dataset: list = []

    for i in range(0, n):
        dataset.append((i, i*2))

    return dataset

def initialize_weights(x,y):
    return rng.uniform(x,y,1)

def calculate_loss(model, dataset):
    MSE = 0
    for (x,y) in dataset:
        MSE += (y - x*model)**2
    return MSE    

def train_model(weight, dataset, n_epochs, use_learing_rate = False, print_all = False):
    for i in range(1, n_epochs+1):
        MSE = calculate_loss(weight, dataset)

        if print_all:
            print("Epoch:", i)
            print("MSE:", MSE)

        plus_eps_MSE = calculate_loss(weight+0.001, dataset)
        L = plus_eps_MSE - MSE
        weight -= (L*LEARNING_RATE) if use_learing_rate else L
    print("Final MSE:", MSE)  
    print("Final weight:", weight) 

def task_01():
    print(create_dataset(4))
    print(initialize_weights(0,100))
    print(initialize_weights(0,10))
    #General form of the model:
    # x (input) -> w1(weight/parameter)*x -> y (output)

def task_02():
    dataset = create_dataset(6)
    np.random.default_rng(42)
    weight = initialize_weights(0,10)
    initial_MSE = calculate_loss(weight, dataset)
    plus_eps_times_two_MSE = calculate_loss(weight+0.001*2, dataset)
    plus_eps_MSE = calculate_loss(weight+0.001, dataset)
    minus_eps_times_two_MSE = calculate_loss(weight-0.001*2, dataset)
    minus_eps_MSE = calculate_loss(weight-0.001, dataset)

    print("Initial MSE:", initial_MSE)
    print("w + 0.001 * 2 MSE:", plus_eps_times_two_MSE)
    print("w + 0.001 MSE:", plus_eps_MSE)
    print("w - 0.001 * 2 MSE:", minus_eps_times_two_MSE)
    print("w - 0.001 MSE:", minus_eps_MSE)
    #Loss function is lower when weight is lower, which means that the optimal weight is lower than our current one.

def task_03():
    dataset = create_dataset(6)
    weight = initialize_weights(0,10)
    print("Weight:", weight)

    initial_MSE = calculate_loss(weight, dataset)
    print("Initial MSE:", initial_MSE)

    plus_eps_MSE = calculate_loss(weight+0.001, dataset)
    L = plus_eps_MSE - initial_MSE
    weight_with_update = weight - L
    after_updating_MSE =  calculate_loss(weight_with_update, dataset)
    print("After updating parameter MSE:", after_updating_MSE)

    weight_with_update_learning_rate = weight - LEARNING_RATE*L
    after_updating_MSE_learning_rate = calculate_loss(weight_with_update_learning_rate, dataset)
    print("After updating parameter with learning rate MSE:", after_updating_MSE_learning_rate)

    train_model(weight, dataset, 10)

def task_04():
    dataset = create_dataset(6)
    weight = initialize_weights(0,10)
    train_model(weight, dataset, 500)
    

def main():
    task_04()

if __name__ == "__main__":
    main()
