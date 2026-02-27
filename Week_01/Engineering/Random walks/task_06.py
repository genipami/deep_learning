import numpy as np
import matplotlib.pyplot as plt
from random_walks_helpers import simulate_n_walks

def main():
    np.random.default_rng(seed=123)
    all_walks = simulate_n_walks(5, 100, True, False)
    np_all_walks = np.array(all_walks).T
    plt.plot(np_all_walks)
    plt.xlabel("Throw")
    plt.title("Random walks")
    plt.show()

if __name__ == "__main__":
    main()