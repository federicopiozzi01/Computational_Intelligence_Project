from Problem import Problem
from src.precompute import precompute_D_Db_paths
from src.ga_core import genetic_algorithm

POPULATION_SIZE = 160
NUM_GENERATIONS = 500
MUTATION_RATE = 0.15


def solution(p: Problem):
    """
    Returns the solution path in the required format.
    """
    D, Db, P = precompute_D_Db_paths(p)

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
