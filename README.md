# Computational_Intelligence_Project

## Problem v-0
This notebook implements Simulated Annealing as the primary optimization metaheuristic, enhanced with a custom Reheating Strategy to prevent stagnation in local optima. The performance of this stochastic approach is benchmarked against a deterministic Dijkstra's Algorithm baseline.

## Problem v-1
This notebook solves the optimization task using a population-based Genetic Algorithm. The solver iterates through generations, applying mutation and selection to improve solution quality. The final performance is benchmarked against a standard Dijkstra baseline.

## Problem v-1.1
This notebook features an optimized implementation of the Genetic Algorithm. It introduces a Precomputed Distance Matrix to speed up the cost function evaluation during the evolutionary process. The final solution is benchmarked against a standard Dijkstra baseline.

## Problem v-2
This notebook solves the optimization problem using an Ant Colony System (ACS) Min-Max algorithm. It features a stagnation detection mechanism that triggers a global pheromone reset (restart) to escape local optima. The stochastic performance is validated against a standard Dijkstra baseline.

## Problem v-3
This version integrates the previously developed algorithms into a unified testing environment. It compares the Genetic Algorithm and Ant Colony Optimization against the Dijkstra baseline, producing a final report on which method finds the best solution for the given graph topology.

## Problem v-4
This notebook implements a time-constrained Advanced Physics Search algorithm. It combines a stochastic exploration phase with a deterministic Polishing (Local Search) mechanism to refine solutions. The solver iterates continuously within a set time limit to discover and improve the best found path.

## Problem v-5
This notebook features a sophisticated Genetic Algorithm that adapts to the problem constraints. It employs Hybrid Initialization (combining Greedy and Clustering heuristics) and Structural Mutations to explore complex path topologies, including multi-trip solutions (intermediate returns to the start node).

## Problem v-6
This notebook upgrades the Genetic Algorithm with a Deterministic Crowding mechanism to maintain population diversity. It combines Hill Climbing initialization strategies with $\beta$-specific genetic operators to robustly explore the search space avoiding local optima.

## Problem v-7
This notebook implements a Genetic Algorithm focused on optimizing the trip structure. It features a novel Zero-Preserving Crossover that allows the solver to learn the optimal number of returns to the depot from the best individuals, while standard mutations refine the city visiting order.

## Problem v-8
This notebook implements a state-of-the-art Hybrid Solver. It alternates between a Genetic Algorithm (optimizing city order) and a Hill Climbing Refiner (optimizing trip structure). It features a "Smart Evaluation" mechanism that automatically decides the best points to return to the depot, allowing the solver to iteratively boost performance by feeding polished results back into the population.

## Problem v-9
This notebook acts as a "Grand Prix" for initialization strategies. It implements sophisticated constructive algorithms—including Clarke & Wright, Regret Insertion, and Cost-Aware Greedy—and runs comparative benchmarks to determine the optimal way to seed the Genetic Algorithm's population for different problem scenarios ($\beta$ regimes).

## Problem v-10
This notebook presents the final, optimized version of the solver. It features a Hybrid Genetic Search (HGS) powered by Numba JIT compilation. By moving the computational bottlenecks (Cost evaluation, Split, and Local Search) to compiled code, this version handles larger populations and more generations in a fraction of the time, allowing for deeper exploration of the search space.