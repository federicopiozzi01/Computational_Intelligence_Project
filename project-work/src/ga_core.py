import random
from Problem import Problem
from src.cost import get_Cost
from src.output_builder import build_output_path


def init_population(problem: Problem, population_size: int):
    """
    Initialize a GA population as a list of random permutations of the cities.
    """
    cities = list(range(1, len(problem._graph.nodes)))
    population = []
    for _ in range(population_size):
        ind = cities.copy()
        random.shuffle(ind)
        population.append(ind)
    return population


def tournament_selection(problem, population, D, Db, tournament_size=3):
    """
    Tournament selection (minimization): pick `tournament_size` random individuals
    and return the one with lowest route cost.
    """
    candidates = random.sample(population, tournament_size)
    best = None
    best_cost = float("inf")

    for c in candidates:
        cost = get_Cost(problem, c, D, Db)
        if cost < best_cost:
            best_cost = cost
            best = c
    return best


def tweek(solution):
    """
    Small local perturbation (swap mutation):
    randomly swaps two positions in the permutation
    """
    i, j = random.sample(range(len(solution)), 2)
    sol = solution.copy()
    sol[i], sol[j] = sol[j], sol[i]
    return sol


def mutation(individual, mutation_rate=0.1):
    """
    Apply mutation with probability `mutation_rate`.n
    """
    return tweek(individual) if random.random() < mutation_rate else individual


def crossover(p1, p2):
    """
     Order-based crossover for permutation
    """
    size = len(p1)
    a, b = sorted(random.sample(range(size), 2))
    child = [None] * size
    # Inherit a slice from parent1
    child[a : b + 1] = p1[a : b + 1]
    # Fill remaining slots with genes from parent2 in order
    fill = [x for x in p2 if x not in child]
    idx = 0
    for i in range(size):
        if child[i] is None:
            child[i] = fill[idx]
            idx += 1
    return child


def genetic_algorithm(
    problem: Problem,
    D,
    Db,
    P,
    population_size=160,
    generations=500,
    mutation_rate=0.15,
):
    """
    Main GA loop to optimize the visiting order of cities.
    
    Key design choices:
    - Fitness evaluation uses `get_Cost(...)`, which already models the
      load-dependent travel cost and a greedy unload-to-depot decision.
    - Elitism: the best individual of each generation is carried over unchanged,
      guaranteeing that the best-found solution is never lost.
    - Parent selection: tournament selection.
    - Recombination: order-preserving crossover
    - Mutation: swap mutation.

     Parameters
    ----------
    D, Db:
        Precomputed matrices used by the cost function for fast evaluation.
    P:
        Additional precomputed data passed to `build_output_path` 
        to reconstruct the explicit route (including intermediate nodes / depot).
    """
    population = init_population(problem, population_size)
    best_solution = None
    best_cost = float("inf")

    for _ in range(generations):
        # Sort population so that the first element is the current elite
        population.sort(key=lambda x: get_Cost(problem, x, D, Db))
        elite = population[0]
        
        # Track global best solution seen so far (not only generation best)
        cost = get_Cost(problem, elite, D, Db)
        if cost < best_cost:
            best_cost = cost
            best_solution = elite.copy()

         # Elitism: keep the elite unchanged in the next generation
        new_population = [elite]

        # Fill the rest of the new population with offspring
        while len(new_population) < population_size:
            p1 = tournament_selection(problem, population, D, Db)
            p2 = tournament_selection(problem, population, D, Db)
            # Create a feasible child permutation via crossover + mutation
            child = mutation(crossover(p1, p2), mutation_rate)
            new_population.append(child)

        population = new_population
    # Convert the best permutation into the required output path format
    return build_output_path(problem, best_solution, D, Db, P), best_cost

