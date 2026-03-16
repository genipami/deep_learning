import numpy as np
from random_walks_helpers import simulate_roll

def main():
    np.random.seed(123)
    floating = np.random.rand()
    first_int = np.random.randint(1,7,1)[0]
    second_int = np.random.randint(1,7,1)[0]
    step = 50

    print("Random float:", floating)
    print("Random integer 1:", first_int)
    print("Random integer 2:", second_int)
    print("Before throw step =", step)

    (roll, step) = simulate_roll(step)

    print("After throw dice =", roll)
    print("After throw step =", step)

if __name__ == "__main__":
    main()
