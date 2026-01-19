import logging
import time
import random
import numpy as np
import networkx as nx
from Problem import Problem

# =========================
# Costanti GA
# =========================
POPULATION_SIZE = 100
NUM_GENERATIONS = 1000  
MUTATION_RATE = 0.3     # Aumentato leggermente per evitare minimi locali
TOURNAMENT_SIZE = 5     # Aumentato per maggiore pressione selettiva

# =========================
# Precompute distances
# =========================
def precompute_distances(problem):
    """
    Calcola matrice delle distanze tra tutte le coppie di nodi nel grafo.
    """
    graph = problem._graph
    n = len(graph.nodes)
    dist_matrix = np.zeros((n, n))
    
    # Usa Floyd-Warshall o Dijkstra ripetuto. Per grafi sparsi, Dijkstra è ok.
    # Nota: nx.all_pairs_dijkstra_path_length restituisce generatori
    path_lengths = dict(nx.all_pairs_dijkstra_path_length(graph, weight="dist"))
    
    for i in range(n):
        for j in range(n):
            if j in path_lengths[i]:
                dist_matrix[i][j] = path_lengths[i][j]
            else:
                dist_matrix[i][j] = float('inf') # Non connesso (non dovrebbe succedere)
    
    return dist_matrix

# =========================
# Popolazione iniziale
# =========================
def init_population(problem: Problem, population_size: int, distance_matrix) -> list[list[int]]:
    population = []
    nodes = list(problem._graph.nodes)
    if 0 in nodes:
        nodes.remove(0)
    
    # 1. Individuo Greedy (Nearest Neighbor)
    current_node = 0
    unvisited = set(nodes)
    greedy_ind = []
    while unvisited:
        next_node = min(unvisited, key=lambda x: distance_matrix[current_node][x])
        greedy_ind.append(next_node)
        unvisited.remove(next_node)
        current_node = next_node
    population.append(greedy_ind)

    # 2. Individui casuali
    for _ in range(population_size - 1):
        individual = nodes.copy()
        random.shuffle(individual)
        population.append(individual)
    
    return population

# =========================
# Cost function (Fast Proxy)
# =========================
def getFastSmartCost(problem: Problem, solution: list[int], dist_matrix) -> float:
    """
    Calcola una stima veloce del costo.
    NOTA: Questa funzione usa la distanza diretta (somma dei pesi degli archi).
    Se beta != 1, matematicamente (d1+d2)^beta != d1^beta + d2^beta.
    Tuttavia, per l'evoluzione genetica, questa approssimazione è sufficiente 
    e molto più veloce di calcolare il path reale nodo per nodo.
    """
    cost = 0.0
    weight = 0.0
    current_node = 0
    
    alpha = problem._alpha
    beta = problem._beta

    for next_node in solution:
        d_direct = dist_matrix[current_node][next_node]
        d_via_0 = dist_matrix[current_node][0] + dist_matrix[0][next_node]
        
        # Costo stimato per andare diretto
        # Assumiamo che il path sia un unico segmento lungo d_direct
        c_direct = d_direct + (alpha * d_direct * weight) ** beta
        
        # Costo stimato via 0 (scarico)
        # 1. current -> 0 (con peso attuale)
        c_to_0 = dist_matrix[current_node][0] + (alpha * dist_matrix[current_node][0] * weight) ** beta
        # 2. 0 -> next (peso 0)
        c_from_0 = dist_matrix[0][next_node] # + 0 penalty
        
        c_via_0 = c_to_0 + c_from_0

        if c_via_0 < c_direct:
            cost += c_via_0
            weight = 0.0
        else:
            cost += c_direct
        
        weight += problem._graph.nodes[next_node].get("gold", 1.0)
        current_node = next_node

    # Ritorno finale a 0
    d_home = dist_matrix[current_node][0]
    cost += d_home + (alpha * d_home * weight) ** beta
    
    return cost

# =========================
# Operatori GA
# =========================
def tournament_selection(problem, population, dist_matrix, size=3):
    candidates = random.sample(population, size)
    best = min(candidates, key=lambda x: getFastSmartCost(problem, x, dist_matrix))
    return best

def crossover_ox(parent1, parent2):
    size = len(parent1)
    a, b = sorted(random.sample(range(size), 2))
    child = [None] * size
    child[a:b+1] = parent1[a:b+1]
    
    pos = 0
    for gene in parent2:
        if gene not in child[a:b+1]:
            while child[pos] is not None:
                pos += 1
            child[pos] = gene
    return child

def mutate(individual, mutation_rate):
    if random.random() < mutation_rate:
        # Swap mutation
        idx1, idx2 = random.sample(range(len(individual)), 2)
        individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
    return individual

# =========================
# Algoritmo Genetico
# =========================
def genetic_algorithm(problem, dist_matrix):
    population = init_population(problem, POPULATION_SIZE, dist_matrix)
    
    best_solution = min(population, key=lambda x: getFastSmartCost(problem, x, dist_matrix))
    best_fitness = getFastSmartCost(problem, best_solution, dist_matrix)
    
    stagnation_counter = 0

    print("Inizio evoluzione...") # <--- STAMPA DI AVVIO

    for gen in range(NUM_GENERATIONS):
        # --- AGGIUNGI QUESTO BLOCCO ---
        if gen % 100 == 0:
            print(f"Generazione {gen}/{NUM_GENERATIONS} - Best Fitness: {best_fitness:.2f}")
        # ------------------------------

        new_population = []
        
        # Elitismo
        population.sort(key=lambda x: getFastSmartCost(problem, x, dist_matrix))
        new_population.append(population[0])
        
        # ... (resto del codice uguale) ...
        current_best_fitness = getFastSmartCost(problem, population[0], dist_matrix)
        if current_best_fitness < best_fitness:
            best_fitness = current_best_fitness
            best_solution = population[0]
            stagnation_counter = 0
        else:
            stagnation_counter += 1

        current_mutation_rate = MUTATION_RATE
        if stagnation_counter > 50:
            current_mutation_rate = 0.5 
        
        while len(new_population) < POPULATION_SIZE:
            p1 = tournament_selection(problem, population, dist_matrix, TOURNAMENT_SIZE)
            p2 = tournament_selection(problem, population, dist_matrix, TOURNAMENT_SIZE)
            child = crossover_ox(p1, p2)
            child = mutate(child, current_mutation_rate)
            new_population.append(child)
            
        population = new_population
    
    print("Evoluzione completata!") # <--- STAMPA DI FINE
    return best_solution
# =========================
# Costruzione Path Finale e Calcolo "Vero" Costo
# =========================
def build_final_path_and_check(problem: Problem, permutation: list[int], dist_matrix):
    """
    Costruisce il path formattato [(nodo, oro), ...] inserendo i ritorni a 0 ottimali.
    Inoltre calcola il costo ESATTO usando problem.cost() sui segmenti reali.
    """
    alpha = problem._alpha
    beta = problem._beta
    
    formatted_path = []
    current_node = 0
    weight = 0.0
    
    # Variabile per tracciare il costo reale per verifica
    true_cost_accumulated = 0.0

    for next_node in permutation:
        # Calcoli euristici per decidere se passare da 0
        d_direct = dist_matrix[current_node][next_node]
        c_direct = d_direct + (alpha * d_direct * weight) ** beta
        
        c_via_0 = (dist_matrix[current_node][0] + (alpha * dist_matrix[current_node][0] * weight) ** beta) + \
                  dist_matrix[0][next_node]

        # Decisione
        if c_via_0 < c_direct:
            if current_node != 0:
                # Eseguiamo il ritorno a 0 reale
                path_segment = nx.shortest_path(problem._graph, current_node, 0, weight='dist')
                true_cost_accumulated += problem.cost(path_segment, weight)
                formatted_path.append((0, 0))
                
            weight = 0.0
            current_node = 0

        # Muovi verso next_node
        path_segment = nx.shortest_path(problem._graph, current_node, next_node, weight='dist')
        true_cost_accumulated += problem.cost(path_segment, weight)
        
        gold = problem._graph.nodes[next_node].get('gold', 0)
        formatted_path.append((next_node, gold))
        weight += gold
        current_node = next_node

    # Ritorno finale a 0
    if current_node != 0:
        path_segment = nx.shortest_path(problem._graph, current_node, 0, weight='dist')
        true_cost_accumulated += problem.cost(path_segment, weight)
        formatted_path.append((0, 0))

    return formatted_path, true_cost_accumulated

# =========================
# Entry Point
# =========================
def solution(p: Problem):
    # 1. Precomputazione
    dist_matrix = precompute_distances(p)
    
    # 2. GA per trovare la permutazione migliore
    best_perm = genetic_algorithm(p, dist_matrix)
    
    # 3. Costruzione della soluzione formattata e calcolo costo preciso
    final_path, real_cost = build_final_path_and_check(p, best_perm, dist_matrix)
    
    # Se vuoi stampare il costo per debug:
    # print(f"GA Real Cost: {real_cost}")
    
    return final_path

# =========================
# Test Locale
# =========================
if __name__ == "__main__":
    # Configurazione identica a Problem.py defaults o custom
    p = Problem(100, density=1, alpha=1, beta=2) 
    
    
    
    start_time = time.time()
    sol = solution(p)
    end_time = time.time()
    
    # Verifica validità formato
    print("\n=== Formato Soluzione ===")
    print(sol[:5], "...", sol[-1])
    
    # Confronto con Baseline
    print("\n=== Confronto Costi ===")
    
    # 1. Calcolo costo Soluzione GA usando rigorosamente Problem.cost
    # Dobbiamo ricostruire i path passo-passo perché 'sol' contiene solo i "salti"
    ga_cost_check = 0
    current = 0
    current_w = 0
    for node, gold in sol:
        # Trova shortest path nel grafo reale
        path = nx.shortest_path(p._graph, current, node, weight='dist')
        # Applica formula ufficiale
        ga_cost_check += p.cost(path, current_w)
        # Aggiorna stato
        if node == 0:
            current_w = 0
        else:
            current_w += gold
        current = node
        
    print(f"GA Cost (Verified): {ga_cost_check:.4f}")
    
    # 2. Baseline Cost
    base_cost = p.baseline()
    print(f"Baseline Cost:      {base_cost:.4f}")
    
    print(f"Improvement:        {((base_cost - ga_cost_check)/base_cost)*100:.2f}%")
    print(f"Time elapsed:       {end_time - start_time:.4f}s")
