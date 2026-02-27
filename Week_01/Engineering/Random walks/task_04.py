import numpy as np
import matplotlib.pyplot as plt
from random_walks_helpers import simulate_walk

def main():
    np.random.default_rng(seed=123)
    walk = simulate_walk(100, True)
    plt.plot(walk)
    plt.xlabel("Throw")
    plt.title("Random walk")
    plt.show()

if __name__ == "__main__":
    main()