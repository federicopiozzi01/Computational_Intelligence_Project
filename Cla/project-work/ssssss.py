from Problem import Problem

import time
import random
import numpy as np
import networkx as nx


# =========================
# Costanti GA 
# =========================
POPULATION_SIZE = 100
NUM_GENERATIONS = 1000
MUTATION_RATE = 0.2


# =========================
# Precompute distances
# =========================
def precompute_D_and_Db(problem: Problem):
    """
    Precompute per ogni coppia (i,j):
    - D[i][j]  = somma delle distanze degli archi sul cammino minimo i->j
    - Db[i][j] = somma (dist_edge ** beta) degli archi sul cammino minimo i->j

    Così il costo coerente edge-by-edge diventa O(1):
    segment_cost(i,j,w) = D[i][j] + (alpha*w)^beta * Db[i][j]
    """
    g = problem._graph
    n = len(g.nodes)
    beta = problem._beta

    D = np.full((n, n), np.inf, dtype=float)
    Db = np.zeros((n, n), dtype=float)
    # Distance from a node to itself is always 0
    for i in range(n):
        D[i][i] = 0.0
        Db[i][i] = 0.0

    # per ogni sorgente calcolo distanze e path minimi
    for src in range(n):
        # path_dict[tgt] is the list of nodes on the shortest path src -> tgt
        dist_dict, path_dict = nx.single_source_dijkstra(g, src, weight="dist")
        for tgt, path in path_dict.items():
            # sommo gli archi del path
            d_sum = 0.0
            db_sum = 0.0
            for u, v in zip(path, path[1:]):
                d = g[u][v]["dist"]
                d_sum += d
                db_sum += d ** beta
            D[src][tgt] = d_sum
            Db[src][tgt] = db_sum

    return D, Db

# =========================
# Cost function (coerente edge-by-edge con segment cost O(1))
# =========================
def segment_cost_fast(problem: Problem, a: int, b: int, weight: float, D, Db) -> float:
    """
    Costo coerente edge-by-edge sul cammino minimo a->b, ma calcolato in O(1):
    sum(d_e) + (alpha*weight)^beta * sum(d_e^beta)
    """
    if a == b:
        return 0.0
    alpha = problem._alpha
    beta = problem._beta
    return D[a][b] + ((alpha * weight) ** beta) * Db[a][b]


def getCoherentCost(problem: Problem, solution: list[int], D, Db) -> float:
    """
    Fitness coerente con il problema (robusta per beta > 1),
    mantenendo la tua scelta greedy "diretto vs via 0", ma con segment cost O(1).
    """
    cost = 0.0
    weight = 0.0
    current_node = 0

    for next_node in solution:
        cost_direct = segment_cost_fast(problem, current_node, next_node, weight, D, Db)
        cost_via_0 = (
            segment_cost_fast(problem, current_node, 0, weight, D, Db)
            + segment_cost_fast(problem, 0, next_node, 0.0, D, Db)
        )

        if cost_via_0 < cost_direct:
            cost += cost_via_0
            weight = 0.0
            current_node = 0
        else:
            cost += cost_direct

        weight += problem._graph.nodes[next_node].get("gold", 1)
        current_node = next_node

    cost += segment_cost_fast(problem, current_node, 0, weight, D, Db)
    return cost

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
# Tournament selection
# =========================
def tournament_selection(
    problem: Problem,
    population: list[list[int]],
    D,
    Db,
    tournament_size: int = 3,
):
    candidates = random.sample(population, tournament_size)

    best_candidate = None
    best_cost = float("inf")

    for cand in candidates:
        cost = getCoherentCost(problem, cand, D, Db)
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
    D,
    Db,
) -> list[tuple[int, float]]:
    path_out: list[tuple[int, float]] = []
    weight = 0.0
    current_node = 0

    for next_node in best_solution:
        cost_direct = segment_cost_fast(problem, current_node, next_node, weight, D, Db)
        cost_via_0 = (
            segment_cost_fast(problem, current_node, 0, weight, D, Db)
            + segment_cost_fast(problem, 0, next_node, 0.0, D, Db)
        )

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
    D,
    Db,
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
        population.sort(key=lambda x: getCoherentCost(problem, x, D, Db))
        best_of_gen = population[0]
        cost_of_gen = getCoherentCost(problem, best_of_gen, D, Db)

        if cost_of_gen < best_cost:
            best_cost = cost_of_gen
            best_solution = best_of_gen.copy()

        new_population.append(best_of_gen)
        # -----------------

        # Riempiamo il resto della nuova popolazione (population_size - 1)
        while len(new_population) < population_size:
            # Parent Selection
            parent1 = tournament_selection(problem, population, D, Db)
            parent2 = tournament_selection(problem, population, D, Db)

            # Crossover
            child = crossover(parent1, parent2)

            # Mutation
            child = mutation(child, mutation_rate)

            new_population.append(child)

        # Sostituiamo la vecchia popolazione con la nuova
        population = new_population

    # ===== INTEGRAZIONE: costruisco il path finale nel formato richiesto =====
    final_path = build_solution_with_depot_visits(problem, best_solution, D, Db)
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
    D, Db = precompute_D_and_Db(p)

    # Se vuoi misurare il tempo puoi farlo qui, ma non è necessario per il grader
    
    sol_ga, cost_ga = genetic_algorithm(
        p,
        D=D,
        Db=Db,
        population_size=POPULATION_SIZE,
        generations=NUM_GENERATIONS,
        mutation_rate=MUTATION_RATE,
    )
    
    return sol_ga

# =========================
# Valutazione del path prodotto da richiamare localmente

def evaluate_output_path(problem: Problem, path_out: list[tuple[int, float]], D, Db) -> float:
    nodes_seq = [0] + [c for c, _ in path_out]
    weight = 0.0
    total = 0.0

    for a, b in zip(nodes_seq, nodes_seq[1:]):
        total += segment_cost_fast(problem, a, b, weight, D, Db)

        if b == 0:
            weight = 0.0
        else:
            weight += problem._graph.nodes[b].get("gold", 1)

    return total


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
    D, Db = precompute_D_and_Db(p)

    print("=== Esecuzione GA ===")
    start = time.time()
    sol = solution(p)
    ga_time = time.time() - start
    

    print("\n=== SOLUZIONE GA ===")
    print("Path:")
    for c, g in sol:
        print(f"({c}, {g:.2f})")


    cost = evaluate_output_path(p, sol, D, Db)
    # 2) Baseline
    t0 = time.time()
    baseline_cost = p.baseline()
    baseline_time = time.time() - t0
    improvement = (baseline_cost - cost) / baseline_cost * 100.0 if baseline_cost != 0 else 0.0

    print("\nCosto totale GA (evaluate_output_path): {:.2f} (in {:.2f} seconds)".format(cost, ga_time))
    print("\nCosto baseline: {:.2f} (in {:.2f} seconds)".format(baseline_cost, baseline_time))
    print("\n Improvement (GA - Baseline): {:.2f}%".format(improvement))

if __name__ == "__main__":
    run_local()
