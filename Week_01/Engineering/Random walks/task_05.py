import numpy as np
from random_walks_helpers import simulate_n_walks

def main():
    np.random.default_rng(seed=123)
    all_walks = simulate_n_walks(5, 100, True, False)
    print(all_walks) 

if __name__ == "__main__":
    main()