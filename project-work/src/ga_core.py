import random
from Problem import Problem
from src.cost import get_Cost
from src.output_builder import build_output_path


def init_population(problem: Problem, population_size: int):
    cities = list(range(1, len(problem._graph.nodes)))
    population = []
    for _ in range(population_size):
        ind = cities.copy()
        random.shuffle(ind)
        population.append(ind)
    return population


def tournament_selection(problem, population, D, Db, tournament_size=3):
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
    i, j = random.sample(range(len(solution)), 2)
    sol = solution.copy()
    sol[i], sol[j] = sol[j], sol[i]
    return sol


def mutation(individual, mutation_rate=0.1):
    return tweek(individual) if random.random() < mutation_rate else individual


def genetic_algorithm(
    problem: Problem,
    D,
    Db,
    P,
    population_size=160,
    generations=500,
    mutation_rate=0.15,
):
    population = init_population(problem, population_size)
    best_solution = None
    best_cost = float("inf")

    for _ in range(generations):
        population.sort(key=lambda x: get_Cost(problem, x, D, Db))
        elite = population[0]

        cost = get_Cost(problem, elite, D, Db)
        if cost < best_cost:
            best_cost = cost
            best_solution = elite.copy()

        new_population = [elite]

        while len(new_population) < population_size:
            p1 = tournament_selection(problem, population, D, Db)
            p2 = tournament_selection(problem, population, D, Db)
            child = mutation(crossover(p1, p2), mutation_rate)
            new_population.append(child)

        population = new_population

    return build_output_path(problem, best_solution, D, Db, P), best_cost


def crossover(p1, p2):
    size = len(p1)
    a, b = sorted(random.sample(range(size), 2))
    child = [None] * size
    child[a : b + 1] = p1[a : b + 1]

    fill = [x for x in p2 if x not in child]
    idx = 0
    for i in range(size):
        if child[i] is None:
            child[i] = fill[idx]
            idx += 1
    return child
