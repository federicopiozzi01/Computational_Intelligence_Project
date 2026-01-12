from Problem import Problem

import time
import random
import numpy as np
import networkx as nx


# =========================
# Costanti GA (come nel notebook)
# =========================
POPULATION_SIZE = 100
NUM_GENERATIONS = 1000
MUTATION_RATE = 0.2


# =========================
# Precompute distances
# =========================
def precompute_distances(problem):
    """
    Calcola matrice delle distanze tra tutte le coppie di nodi nel grafo del problema.
    Restituisce una matrice 2D numpy NxN dove l'elemento (i, j) rappresenta la distanza tra il nodo i e il nodo j.
    """
    graph = problem._graph
    n = len(graph.nodes)

    dist_matrix = np.zeros((n, n))
    all_pairs = nx.all_pairs_dijkstra_path_length(graph, weight="dist")

    for src, targets in all_pairs:
        for tgt, d in targets.items():
            dist_matrix[src][tgt] = d

    return dist_matrix


# =========================
# Popolazione iniziale
# =========================
def init_population(problem: Problem, population_size: int) -> list[list[int]]:
    """
    Inizializza una popolazione di soluzioni casuali.
    Ogni soluzione è una permutazione casuale delle città (escluso il deposito 0).
    """
    num_cities = len(problem._graph.nodes)
    cities = list(range(1, num_cities))  # Escludo il deposito 0

    population = []
    for _ in range(population_size):
        individual = cities.copy()
        random.shuffle(individual)
        population.append(individual)
    return population


# =========================
# Cost function (fast smart)
# =========================
def getFastSmartCost(problem: Problem, solution: list[int], dist_matrix) -> float:
    """
    For each step, calculate the cost considering the accumulated weight using the precomputed distance matrix.
    """
    cost = 0
    weight = 0
    current_node = 0  # start from the node 0

    alpha = problem._alpha
    beta = problem._beta

    # Visit each city in the solution path
    for next_node in solution:
        # get distance from precomputed matrix
        dist_direct = dist_matrix[current_node][next_node]
        dist_to_0 = dist_matrix[current_node][0]
        dist_from_0 = dist_matrix[0][next_node]

        # Aggiorniamo il costo
        cost_direct = dist_direct + (alpha * dist_direct * weight) ** beta

        cost_via_0 = dist_to_0 + (alpha * dist_to_0 * weight) ** beta + dist_from_0
        if cost_via_0 < cost_direct:
            cost += cost_via_0
            weight = 0
        else:
            cost += cost_direct

        weight += problem._graph.nodes[next_node].get("gold", 1)
        current_node = next_node

    dist_final = dist_matrix[current_node][0]
    cost += dist_final + (alpha * dist_final * weight) ** beta

    return cost


# =========================
# Tournament selection
# =========================
def tournament_selection(
    problem: Problem,
    population: list[list[int]],
    distance_matrix,
    tournament_size: int = 3,
):
    # Seleziono k individui casuali dalla popolazione
    candidates = random.sample(population, tournament_size)

    # Valuto chi tra loro a il costo migliore (minore)
    best_candidate = None
    best_cost = float("inf")

    for cand in candidates:
        cost = getFastSmartCost(problem, cand, distance_matrix)
        if cost < best_cost:
            best_cost = cost
            best_candidate = cand
    return best_candidate


# =========================
# Crossover OX
# =========================
def crossover(parent1: list[int], parent2: list[int]) -> list[int]:
    """
    Applica l'operatore di crossover Order Crossover (OX) per generare un figlio
    da due genitori.
    """
    size = len(parent1)

    # Scelgo due punti di taglio casuali
    start, end = sorted(random.sample(range(size), 2))

    # Creo il figlio con None e copiamo la parte centrale da parent1
    child = [None] * size
    child[start : end + 1] = parent1[start : end + 1]  # serve "+1" per includere end

    # Riempio il resto del figlio con gli elementi di parent2 nell'ordine in cui appaiono
    p2_genes = [gene for gene in parent2 if gene not in child[start : end + 1]]

    current_p2_index = 0
    for i in range(size):
        if child[i] is None:
            child[i] = p2_genes[current_p2_index]
            current_p2_index += 1
    return child


# =========================
# Tweak + Mutation
# =========================
def tweak(solution: list[int]) -> list[int]:
    """
    Applia un piccolo cambiamento (tweak) alla soluzione esistente
    scambiando due città nella lista.
    """
    if len(solution) < 2:
        return solution.copy()

    new_solution = solution.copy()
    i, j = random.sample(range(len(solution)), 2)
    new_solution[i], new_solution[j] = new_solution[j], new_solution[i]
    return new_solution


def mutation(individual: list[int], mutation_rate: float = 0.1) -> list[int]:
    """
    Applica la mutazione alla soluzione scambiando due città con una certa probabilità.
    """
    if random.random() < mutation_rate:
        return tweak(individual)
    else:
        return individual


# =========================
# Builder: costruisce path con (0,0) intermedi
# =========================
def build_solution_with_depot_visits(
    problem: Problem,
    best_solution: list[int],
    dist_matrix,
) -> list[tuple[int, float]]:
    """
    Costruisce un path nel formato richiesto, INSERENDO (0,0) anche in mezzo
    quando la logica "via 0" è più conveniente (come in getFastSmartCost).

    - Raccoglie tutto l'oro quando visita una città
    - Quando decide "via 0", inserisce (0,0) prima di andare alla prossima città
    - Termina sempre con (0,0)
    """
    alpha = problem._alpha
    beta = problem._beta

    path_out: list[tuple[int, float]] = []
    weight = 0.0
    current_node = 0

    for next_node in best_solution:
        dist_direct = dist_matrix[current_node][next_node]
        dist_to_0 = dist_matrix[current_node][0]
        dist_from_0 = dist_matrix[0][next_node]

        cost_direct = dist_direct + (alpha * dist_direct * weight) ** beta
        cost_via_0 = dist_to_0 + (alpha * dist_to_0 * weight) ** beta + dist_from_0

        if cost_via_0 < cost_direct:
            if current_node != 0:
                path_out.append((0, 0))
            weight = 0.0
            current_node = 0

        gold_here = problem._graph.nodes[next_node].get("gold", 1)
        path_out.append((next_node, gold_here))

        weight += gold_here
        current_node = next_node

    if len(path_out) == 0 or path_out[-1] != (0, 0):
        path_out.append((0, 0))

    return path_out


# =========================
# Genetic Algorithm (integrata: ritorna già il path)
# =========================
def genetic_algorithm(
    problem: Problem,
    distance_matrix,
    population_size: int = 50,
    generations: int = 1000,
    mutation_rate: float = 0.1,
):
    # Inizializzo la popolazione
    population = init_population(problem, population_size)

    best_solution = None
    best_cost = float("inf")

    for gen in range(generations):
        new_population = []

        # --- ELITISMO ---
        population.sort(key=lambda x: getFastSmartCost(problem, x, distance_matrix))
        best_of_gen = population[0]
        cost_of_gen = getFastSmartCost(problem, best_of_gen, distance_matrix)

        if cost_of_gen < best_cost:
            best_cost = cost_of_gen
            best_solution = best_of_gen.copy()

        new_population.append(best_of_gen)
        # -----------------

        # Riempiamo il resto della nuova popolazione (population_size - 1)
        while len(new_population) < population_size:
            # Parent Selection
            parent1 = tournament_selection(problem, population, distance_matrix)
            parent2 = tournament_selection(problem, population, distance_matrix)

            # Crossover
            child = crossover(parent1, parent2)

            # Mutation
            child = mutation(child, mutation_rate)

            new_population.append(child)

        # Sostituiamo la vecchia popolazione con la nuova
        population = new_population

    # ===== INTEGRAZIONE: costruisco il path finale nel formato richiesto =====
    final_path = build_solution_with_depot_visits(problem, best_solution, distance_matrix)
    return final_path, best_cost


# =========================
# Entry point richiesto dal progetto
# =========================
def solution(p: Problem):
    """
    Deve ritornare un path nel formato:
    [(c1,g1), (c2,g2), ..., (cN,gN), (0,0)]
    (qui possono apparire anche (0,0) intermedi)
    """
    dist_matrix = precompute_distances(p)

    # Se vuoi misurare il tempo puoi farlo qui, ma non è necessario per il grader
    
    sol_ga, cost_ga = genetic_algorithm(
        p,
        distance_matrix=dist_matrix,
        population_size=POPULATION_SIZE,
        generations=NUM_GENERATIONS,
        mutation_rate=MUTATION_RATE,
    )
    
    return sol_ga


def run_local():
    """
    Funzione di test locale:
    - crea il problema
    - esegue la solution()
    - stampa path e costo
    """
    # Parametri di esempio (puoi cambiarli)
    NUM_CITIES = 100
    DENSITY = 0.2
    ALPHA = 1.0
    BETA = 1.0
    SEED = 42

    print("=== Creazione problema ===")
    p = Problem(
        num_cities=NUM_CITIES,
        density=DENSITY,
        alpha=ALPHA,
        beta=BETA,
        seed=SEED,
    )

    print("=== Precompute distances ===")
    dist_matrix = precompute_distances(p)

    print("=== Esecuzione GA ===")
    start = time.time()
    sol = solution(p)
    ga_time = time.time() - start
    

    print("\n=== SOLUZIONE GA ===")
    print("Path:")
    for c, g in sol:
        print(f"({c}, {g:.2f})")


    # Ricavo la permutazione delle città (escludendo 0)
    perm = [c for c, g in sol if c != 0]

    cost = getFastSmartCost(p, perm, dist_matrix)
    # 2) Baseline
    t0 = time.time()
    baseline_cost = p.baseline()
    baseline_time = time.time() - t0
    improvement = (baseline_cost - cost) / baseline_cost * 100.0 if baseline_cost != 0 else 0.0

    print("\nCosto totale GA (getFastSmartCost): {:.2f} (in {:.2f} seconds)".format(cost, ga_time))
    print("\nCosto baseline: {:.2f} (in {:.2f} seconds)".format(baseline_cost, baseline_time))
    print("\n Improvement (GA - Baseline): {:.2f}%".format(improvement))

if __name__ == "__main__":
    run_local()
