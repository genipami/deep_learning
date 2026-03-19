import numpy as np
import matplotlib.pyplot as plt

EPS = 0.001
LEARNING_RATE = 0.001


def create_dataset(func) -> list:
    dataset: list = []

    for i in range(2):
        for j in range(2):
            dataset.append((i, j, int(func(i, j))))

    return dataset


def initialize_weights(x: float, y: float) -> float:
    return np.random.uniform(x, y)


def sigmoid(x: float) -> float:
    return 1 / (1 + np.exp(-x))


def calculate_loss(model: list, dataset: list) -> float:
    MSE = 0
    for (x1, x2, y) in dataset:
        activation = sigmoid(x1 * model[0] + x2 * model[1] + model[2])

        MSE += (y - activation)**2
    return MSE


def train_model(model: list,
                dataset: list,
                n_epochs: int,
                use_learing_rate: bool = False,
                print_all: bool = False) -> list:
    losses = []

    for i in range(1, n_epochs + 1):
        MSE = calculate_loss(model, dataset)
        losses.append(MSE)

        if print_all:
            print("Epoch:", i)
            print("MSE:", MSE)

        for j in range(3):
            altered_model = np.copy(model)
            altered_model[j] += EPS
            plus_eps_MSE = calculate_loss(altered_model, dataset)
            L = (plus_eps_MSE - MSE) / EPS
            model[j] -= (L * LEARNING_RATE) if use_learing_rate else L

        if print_all:
            print("Final MSE:", MSE)
            for (i, param) in enumerate(model):
                print(f"Parameter {i}: {param}")

    return losses


#General form of OR and AND neural networks:
#There are 2 inputs X1 and X2 and each one has a weight. Also there is one bias. We use the sigmoid function to contain the output values in the range [0,1]
def print_results(model: list, dataset: list) -> None:
    for (x1, x2, y) in dataset:
        print("Expected output:", y, "Prediction:",
              sigmoid(x1 * model[0] + x2 * model[1] + model[2]))


def create_single_layer_bool_nn(func) -> tuple:
    dataset = create_dataset(func)
    model = []

    for i in range(3):
        model.append(initialize_weights(-1, 1))

    losses = train_model(model, dataset, 10000, True)

    return (model, losses)


def plot_losses(losses):
    plt.plot(losses)
    plt.title("Change in MSE over 10 000 epochs")
    plt.xlabel("Epochs")
    plt.ylabel("MSE")
    plt.show()


def forward(model: list, inputs: tuple):
    (x1, x2) = inputs
    return sigmoid(x1 * model[0] + x2 * model[1] + model[2])


def create_xor_nn() -> tuple:
    (or_model, _) = create_single_layer_bool_nn(lambda x, y: x or y)
    (nand_model, _) = create_single_layer_bool_nn(lambda x, y: not (x and y))
    (and_model, _) = create_single_layer_bool_nn(lambda x, y: x and y)

    return (or_model, nand_model, and_model)


def xor_nn_forward(xor_model: tuple, inputs: tuple):
    (or_model, nand_model, and_model) = xor_model

    or_result = np.round(forward(or_model, inputs))
    nand_result = np.round(forward(nand_model, inputs))

    return forward(and_model, (or_result, nand_result))

def main():
    xor_dataset = create_dataset(lambda x, y: (x or y) and not (x and y))
    xor_model = create_xor_nn()
    for (x1, x2, y) in xor_dataset:
        print("Expected output:", y, "Prediction:",
            xor_nn_forward(xor_model, (x1, x2)))


if __name__ == "__main__":
    main()
