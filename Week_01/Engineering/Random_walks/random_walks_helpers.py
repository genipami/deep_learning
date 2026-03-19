import numpy as np

def simulate_roll(current_step:int, should_clamp:bool = True, is_clumsy:bool=False) -> tuple:
    roll = np.random.randint(1,7,1)[0]

    while roll == 6:
        roll = np.random.randint(1,7,1)[0]

    if roll in [1,2]:
        current_step -= 1
    else:
        current_step += 1

    if should_clamp and current_step < 0:
        current_step = 0    

    if is_clumsy:
        clumsiness = np.random.rand()
        if clumsiness <= 0.005:
            current_step = 0

    return (roll, current_step)

def simulate_walk(num_throws: int, should_clamp:bool = True, is_clumsy:bool=False) -> list:
    walk: list = []
    step = 0
    for i in range(0, num_throws):
        walk.append(step)
        (roll, step) = simulate_roll(step, should_clamp, is_clumsy)

    return walk

def simulate_n_walks(num_walks:int, num_throws:int, should_clamp:bool = True, is_clumsy:bool=False)  -> list:
    np.random.default_rng(seed=123)
    all_walks: list = []

    for i in range(0,num_walks):
        walk = simulate_walk(num_throws, should_clamp, is_clumsy)
        all_walks.append(walk)
    
    return all_walks


