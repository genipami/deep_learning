import numpy as np
import matplotlib.pyplot as plt
from random_walks_helpers import simulate_n_walks

def main():
    np.random.seed(123)
    all_walks = simulate_n_walks(500, 100, True, True)
    np_all_walks = np.array(all_walks)
    ending_steps = np_all_walks[:,-1]
    odds = len(ending_steps[ending_steps >= 60]) / len(ending_steps)
    print(odds)
    plt.hist(ending_steps)
    plt.xlabel("End step")
    plt.title("Random walks")
    plt.show()

if __name__ == "__main__":
    main()