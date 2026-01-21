from Problem import Problem
from src.precompute import precompute_D_Db_paths
from src.ga_core import genetic_algorithm

# --- Genetic Algorithm parameters ---
# Chosen empirically to balance solution quality and execution time
POPULATION_SIZE = 160
NUM_GENERATIONS = 500
MUTATION_RATE = 0.15


def solution(p: Problem):
    """
    This function receives an instance of `Problem` and returns a path
    in the required format:
        [(c1, g1), (c2, g2), ..., (cN, gN), (0, 0)]

    """
    D, Db, P = precompute_D_Db_paths(p)

    # Run the Genetic Algorithm to find the best city permutation.
    # The GA returns:
    # `path`: the full route including depot returns and collected gold
    # `_`   : the total cost (ignored here, since only the path is required)
    path, _ = genetic_algorithm(
        p,
        D=D,
        Db=Db,
        P=P,
        population_size=POPULATION_SIZE,
        generations=NUM_GENERATIONS,
        mutation_rate=MUTATION_RATE,
    )

    return path
