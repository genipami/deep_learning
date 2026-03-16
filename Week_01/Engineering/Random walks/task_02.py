import numpy as np
from random_walks_helpers import simulate_walk

def main():
    np.random.seed(123)
    walk = simulate_walk(100, False)
    print(walk)    
    #we get negative steps, which should be impossible 

if __name__ == "__main__":
    main()