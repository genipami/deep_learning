import numpy as np
from random_walks_helpers import simulate_walk

def main():
    np.random.default_rng(seed=123)
    walk = simulate_walk(100, True)
    print(walk)

if __name__ == "__main__":
    main()