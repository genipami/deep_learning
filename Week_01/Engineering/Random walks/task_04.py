import matplotlib.pyplot as plt
from random_walks_helpers import simulate_walk

def main():
    walk = simulate_walk(100, True)
    plt.plot(walk)
    plt.xlabel("Throw")
    plt.title("Random walk")
    plt.show()

if __name__ == "__main__":
    main()