from Problem import Problem

import time
import random
import numpy as np
import networkx as nx


# =========================
# Costanti GA
# =========================
POPULATION_SIZE = 160
NUM_GENERATIONS = 500
MUTATION_RATE = 0.15


# =========================
# Precompute distances + paths
# =========================
def precompute_D_Db_and_paths(problem: Problem):
    """
    Precompute:
      - D[i][j]  = shortest-path distance (sum of edge dist) from i to j
      - Db[i][j] = sum(dist_edge ** beta) along the same shortest path
      - P[i][j]  = list of nodes on the shortest path i -> j (inclusive)
    """
    g = problem._graph
    n = len(g.nodes)
    beta = problem._beta

    D = np.full((n, n), np.inf, dtype=float)
    Db = np.zeros((n, n), dtype=float)

    # P[i][j] is a list of nodes (path) or None
    P: list[list[list[int] | None]] = [[None for _ in range(n)] for _ in range(n)]

    for i in range(n):
        D[i][i] = 0.0
        Db[i][i] = 0.0
        P[i][i] = [i]

    for src in range(n):
        dist_dict, path_dict = nx.single_source_dijkstra(g, src, weight="dist")

        for tgt, path in path_dict.items():
            D[src][tgt] = float(dist_dict[tgt])
            P[src][tgt] = path

            db_sum = 0.0
            for u, v in zip(path, path[1:]):
                d = g[u][v]["dist"]
                db_sum += d ** beta
            Db[src][tgt] = db_sum

    return D, Db, P


# =========================
# Segment cost (coerente edge-by-edge, O(1) grazie a D/Db)
# =========================
def segment_cost_fast(problem: Problem, a: int, b: int, weight: float, D, Db) -> float:
    """
    Costo coerente edge-by-edge sul cammino minimo a->b, calcolato in O(1):
      sum(d_e) + (alpha*weight)^beta * sum(d_e^beta)
    """
    if a == b:
        return 0.0
    alpha = problem._alpha
    beta = problem._beta
    return D[a][b] + ((alpha * weight) ** beta) * Db[a][b]


def getCoherentCost(problem: Problem, solution: list[int], D, Db) -> float:
    """
    Fitness coerente col problema (robusta per beta > 1) usando greedy:
    ad ogni step decide se andare diretto o passare da 0 per scaricare.
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
    cities = list(range(1, num_cities))  # escludo il deposito 0

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
    Order Crossover (OX).
    """
    size = len(parent1)
    start, end = sorted(random.sample(range(size), 2))

    child = [None] * size
    child[start : end + 1] = parent1[start : end + 1]

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
    Swap di due città.
    """
    if len(solution) < 2:
        return solution.copy()

    new_solution = solution.copy()
    i, j = random.sample(range(len(solution)), 2)
    new_solution[i], new_solution[j] = new_solution[j], new_solution[i]
    return new_solution


def mutation(individual: list[int], mutation_rate: float = 0.1) -> list[int]:
    """
    Mutazione con probabilità mutation_rate (swap).
    """
    if random.random() < mutation_rate:
        return tweak(individual)
    return individual


# =========================
# Builder output path (0 iniziale implicito, 0 intermedi e finale espliciti)
# =========================
def append_shortest_path_segment(
    output_path: list[tuple[int, float]],
    shortest_path: list[int],
    *,
    include_start_node: bool,
    pickup_at_end: float,
):
    """
    Appende un segmento di shortest path al path di output.

    - Nodi intermedi: (node, 0.0)
    - Nodo finale: (end, pickup_at_end)
    - include_start_node=False: salta il primo nodo del segmento (evita duplicati)

    Robustezza:
    - evita duplicati consecutivi di nodo
    - se lo stesso nodo appare due volte di fila con pickup diversi, li fonde (tiene il max pickup)
    """
    if not shortest_path:
        return

    start_idx = 0 if include_start_node else 1
    if start_idx >= len(shortest_path):
        return

    # intermedi
    for node in shortest_path[start_idx:-1]:
        tup = (node, 0.0)
        if not output_path:
            output_path.append(tup)
        else:
            last_node, last_pick = output_path[-1]
            if last_node == node:
                output_path[-1] = (last_node, max(last_pick, 0.0))
            else:
                output_path.append(tup)

    # destinazione
    end_node = shortest_path[-1]
    end_pick = float(pickup_at_end)

    if not output_path:
        output_path.append((end_node, end_pick))
    else:
        last_node, last_pick = output_path[-1]
        if last_node == end_node:
            output_path[-1] = (last_node, max(last_pick, end_pick))
        else:
            output_path.append((end_node, end_pick))


def build_output_path(problem: Problem, chromosome: list[int], D, Db, P):
    """
    Costruisce il path nel formato richiesto:

    - (0,0) iniziale: IMPLICITO (non viene inserito)
    - (0,0) intermedi: DEVONO esserci quando scarichi
    - (0,0) finale: DEVE esserci
    - Tutti i nodi attraversati nei shortest paths devono apparire
      (pickup=0.0 se non raccogli oro lì)
    """
    if chromosome is None or len(chromosome) == 0:
        return [(0, 0.0)]

    out: list[tuple[int, float]] = []
    weight = 0.0
    current = 0

    for nxt in chromosome:
        direct = segment_cost_fast(problem, current, nxt, weight, D, Db)
        via_0 = (
            segment_cost_fast(problem, current, 0, weight, D, Db)
            + segment_cost_fast(problem, 0, nxt, 0.0, D, Db)
        )

        if via_0 < direct:
            # vai current -> 0 e inserisci (0,0)
            if current != 0:
                append_shortest_path_segment(
                    out,
                    P[current][0],
                    include_start_node=False,
                    pickup_at_end=0.0,
                )
            else:
                if not out or out[-1][0] != 0:
                    out.append((0, 0.0))

            weight = 0.0
            current = 0

        # vai current -> nxt (non stampare lo 0 iniziale implicito)
        gold = float(problem._graph.nodes[nxt].get("gold", 1))
        include_start = not (current == 0 and len(out) == 0)

        append_shortest_path_segment(
            out,
            P[current][nxt],
            include_start_node=include_start,
            pickup_at_end=gold,
        )

        weight += gold
        current = nxt

    # ritorno finale a 0: deve comparire (0,0)
    if current != 0:
        append_shortest_path_segment(
            out,
            P[current][0],
            include_start_node=False,
            pickup_at_end=0.0,
        )
    else:
        if not out or out[-1][0] != 0:
            out.append((0, 0.0))

    if not out or out[-1] != (0, 0.0):
        if not out or out[-1][0] != 0:
            out.append((0, 0.0))
        else:
            out[-1] = (0, 0.0)

    return out


# =========================
# Cost of FINAL OUTPUT PATH (non dal cromosoma)
# =========================
def compute_output_path_cost(problem: Problem, path: list[tuple[int, float]], D, Db) -> float:
    """
    Calcola il costo del PATH finale prodotto (lista di tuple (city, pickup)).

    Regole rispettate:
    - lo start al deposito (0) è IMPLICITO per il path, ma per il calcolo partiamo da current=0
    - quando arrivi a 0 il peso si resetta a 0 (scarico)
    - il costo del movimento a->b usa segment_cost_fast con il peso corrente PRIMA di muoversi
    - il gold raccolto viene aggiunto DOPO essere arrivati al nodo b (se b != 0)
    """
    if path is None or len(path) == 0:
        return 0.0

    total = 0.0
    current = 0
    weight = 0.0

    for node, pickup in path:
        node = int(node)

        # costo spostamento current -> node con peso attuale
        total += segment_cost_fast(problem, current, node, weight, D, Db)

        # arrivo al nodo
        if node == 0:
            weight = 0.0  # scarico
        else:
            # nel tuo output i nodi intermedi hanno pickup=0.0
            weight += float(pickup)

        current = node

    return total


# =========================
# Genetic Algorithm
# =========================
def genetic_algorithm(
    problem: Problem,
    D,
    Db,
    P,
    population_size: int = 160,
    generations: int = 500,
    mutation_rate: float = 0.15,
):
    population = init_population(problem, population_size)

    best_solution = None
    best_cost = float("inf")

    for _ in range(generations):
        new_population = []

        population.sort(key=lambda x: getCoherentCost(problem, x, D, Db))
        best_of_gen = population[0]
        cost_of_gen = getCoherentCost(problem, best_of_gen, D, Db)

        if cost_of_gen < best_cost:
            best_cost = cost_of_gen
            best_solution = best_of_gen.copy()

        new_population.append(best_of_gen)

        while len(new_population) < population_size:
            parent1 = tournament_selection(problem, population, D, Db)
            parent2 = tournament_selection(problem, population, D, Db)

            child = crossover(parent1, parent2)
            child = mutation(child, mutation_rate)

            new_population.append(child)

        population = new_population

    final_path = build_output_path(problem, best_solution, D, Db, P)
    return final_path


# =========================
# Entry point richiesto dal progetto (qui: restituisco solo path)
# =========================
def solution(p: Problem):
    """
    Deve ritornare un path nel formato:
    [(c1,g1), (c2,g2), ..., (0,0)]
    - 0 iniziale implicito
    - 0 intermedi espliciti quando scarichi
    - 0 finale esplicito
    """
    D, Db, P = precompute_D_Db_and_paths(p)
    sol_path = genetic_algorithm(
        p,
        D=D,
        Db=Db,
        P=P,
        population_size=POPULATION_SIZE,
        generations=NUM_GENERATIONS,
        mutation_rate=MUTATION_RATE,
    )
    return sol_path

# =========================
# Verifiche sul path generato

def check_all_cities_visited(problem: Problem, output_path: list[tuple[int, float]]):
    visited = set(c for c, _ in output_path)
    missing = [i for i in problem._graph.nodes if i != 0 and i not in visited]
    if not missing:
        print("✅ Visit check OK: tutte le città compaiono almeno una volta nel path.")
        return True
    print("❌ ERRORE: queste città non compaiono nel path:", missing)
    return False


def check_pickups(problem: Problem, output_path: list[tuple[int, float]], tol: float = 1e-6):
    """
    Verifica che:
    - non raccogli oro al deposito 0
    - per ogni città i>0, la somma dei pickup nel path sia esattamente problem._graph.nodes[i]['gold']
      (entro tolleranza)
    """
    gold_true = {i: float(problem._graph.nodes[i]["gold"]) for i in problem._graph.nodes if i != 0}
    picked = {i: 0.0 for i in gold_true}

    bad_depot = []
    for (c, g) in output_path:
        g = float(g)
        if c == 0:
            if abs(g) > tol:
                bad_depot.append((c, g))
            continue
        picked[c] += g

    # report
    duplicates = [(i, picked[i], gold_true[i]) for i in picked if picked[i] > gold_true[i] + tol]
    missing    = [(i, picked[i], gold_true[i]) for i in picked if picked[i] < gold_true[i] - tol]

    ok = (len(bad_depot) == 0 and len(duplicates) == 0 and len(missing) == 0)

    if ok:
        print("✅ Pickup check OK: ogni città raccolta esattamente una volta (entro tolleranza).")
        return True

    if bad_depot:
        print("❌ ERRORE: hai pickup != 0 sul deposito 0:", bad_depot[:10], ("..." if len(bad_depot) > 10 else ""))

    if duplicates:
        print("❌ ERRORE: stai raccogliendo TROPPO oro (double pickup) in queste città:")
        for i, got, true in duplicates[:20]:
            print(f"  city {i}: picked={got:.6f}  true={true:.6f}")
        if len(duplicates) > 20:
            print("  ...")

    if missing:
        print("❌ ERRORE: non stai raccogliendo abbastanza oro (manca pickup) in queste città:")
        for i, got, true in missing[:20]:
            print(f"  city {i}: picked={got:.6f}  true={true:.6f}")
        if len(missing) > 20:
            print("  ...")

    return False
def compute_cost_using_problem_cost(problem: Problem, output_path: list[tuple[int, float]]):
    """
    Calcola il costo seguendo ESATTAMENTE Problem.cost([a,b], weight)
    ricostruendo il peso trasportato passo-passo dal path di output.

    Assunzione: output_path contiene anche i nodi intermedi (quindi ogni salto è un edge del grafo).
    """
    total = 0.0
    weight = 0.0
    current = 0

    for nxt, pickup in output_path:
        # costo dell'arco current->nxt con peso corrente
        total += problem.cost([current, nxt], weight)

        # aggiorna peso/pickup
        if nxt == 0:
            weight = 0.0
        else:
            weight += float(pickup)

        current = nxt

    return total
def check_edges_exist(problem: Problem, output_path: list[tuple[int, float]]):
    g = problem._graph
    current = 0
    for nxt, _ in output_path:
        if not g.has_edge(current, nxt):
            print(f"❌ Missing edge: {current} -> {nxt}")
            return False
        current = nxt
    print("✅ Edge check OK: ogni step è un arco del grafo.")
    return True



# =========================
# Run local
# =========================
def run_local():
    NUM_CITIES = 100
    DENSITY = 0.2
    ALPHA = 2.0
    BETA = 3.0
    SEED = 42

    print("=== Creazione problema ===")
    p = Problem(
        num_cities=NUM_CITIES,
        density=DENSITY,
        alpha=ALPHA,
        beta=BETA,
        seed=SEED,
    )

    print("=== Precompute D/Db/P ===")
    D, Db, P = precompute_D_Db_and_paths(p)

    print("=== Esecuzione GA ===")
    start = time.time()
    sol = solution(p)
    ga_time = time.time() - start

    print("\n=== SOLUZIONE GA ===")
    print("Path:")
    for c, g in sol:
        print(f"({c}, {g:.2f})")

    # costo calcolato dal PATH finale (non dal cromosoma)
    cost = compute_output_path_cost(p, sol, D, Db)
    check_all_cities_visited(p, sol)

    check_pickups(p, sol)
    cost_official = compute_cost_using_problem_cost(p, sol)
    print("Costo (Problem.cost edge-by-edge):", round(cost_official, 2))
    check_edges_exist(p, sol)
    # Baseline
    t0 = time.time()
    baseline_cost = p.baseline()
    baseline_time = time.time() - t0
    improvement = (baseline_cost - cost) / baseline_cost * 100.0 if baseline_cost != 0 else 0.0
    print("\n valori inizili: num{}, density={}, alpha={}, beta={}, seed={}".format(NUM_CITIES, DENSITY, ALPHA, BETA, SEED))
    print("\nCosto totale GA (dal PATH finale): {:.2f} (in {:.2f} seconds)".format(cost, ga_time))
    print("Costo baseline: {:.2f} (in {:.2f} seconds)".format(baseline_cost, baseline_time))
    print("Improvement (GA - Baseline): {:.2f}%".format(improvement))


if __name__ == "__main__":
    run_local()
