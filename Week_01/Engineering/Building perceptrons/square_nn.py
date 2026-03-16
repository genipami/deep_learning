import numpy as np
import copy

LEARNING_RATE = 0.0001
EPS = 0.0001


def create_dataset(n: int) -> list:
    dataset: list = []

    for i in range(0, n):
        dataset.append((i, i * i))

    return dataset


def forward(model, x):
    weights, biases = model

    result = np.array([[x]], dtype=float)

    for idx, (weight, bias) in enumerate(zip(weights, biases)):
        result = weight @ result + bias
        if idx < len(weights) - 1:
            result = np.maximum(0, result)

    return result.item()


def calculate_loss(model, dataset):
    MSE = 0
    for (x, y) in dataset:
        MSE += (y - forward(model, x))**2
    MSE /= len(dataset)
    return MSE


def train_model(model,
                dataset,
                n_epochs,
                use_learing_rate=True,
                print_all=False):
    print("MSE before:", calculate_loss(model, dataset))

    for i in range(1, n_epochs + 1):

        MSE = calculate_loss(model, dataset)

        if print_all:
            print("Epoch:", i)
            print("MSE:", MSE)

        (weights, biases) = model
        for i, layer in enumerate(weights):
            for j in range(layer.shape[0]):
                for k in range(layer.shape[1]):
                    altered_model = copy.deepcopy(model)
                    altered_model[0][i][j][k] += EPS

                    plus_eps_MSE = calculate_loss(altered_model, dataset)
                    L = (plus_eps_MSE - MSE) / EPS

                    model[0][i][j][k] -= (
                        L * LEARNING_RATE) if use_learing_rate else L

        for i, layer in enumerate(biases):
            for j in range(layer.shape[0]):
                for k in range(layer.shape[1]):
                    altered_model = copy.deepcopy(model)
                    altered_model[1][i][j][k] += EPS

                    plus_eps_MSE = calculate_loss(altered_model, dataset)
                    L = (plus_eps_MSE - MSE) / EPS

                    model[1][i][j][k] -= (
                        L * LEARNING_RATE) if use_learing_rate else L

    print("Final MSE:", MSE)


def main():
    dataset_size = 100
    dataset = create_dataset(dataset_size)

    n_hidden_neurons = 4

    weights = [
        np.random.uniform(-1, 1, size=(n_hidden_neurons, 1)),
        np.random.uniform(-1, 1, size=(1, n_hidden_neurons))
    ]

    biases = [
        np.random.uniform(-1, 1, size=(n_hidden_neurons, 1)),
        np.random.uniform(-1, 1, size=(1, 1))
    ]

    model = (weights, biases)

    train_model(model, dataset, 10000)


if __name__ == "__main__":
    main()
