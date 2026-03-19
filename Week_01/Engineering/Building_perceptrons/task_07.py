import matplotlib.pyplot as plt
import numpy as np

def sigmoid(x:float) -> float:
    return 1 / (1 + np.exp(-x))

def main():
    xs = np.linspace(-10, 10, 1000)
    ys = list(map(sigmoid, xs))
    plt.plot(xs, ys)
    plt.title("Sigmoid function:")
    plt.show()


if __name__ == "__main__":
    main()