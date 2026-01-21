from Problem import Problem


def compute_segment_cost(problem: Problem, src: int, dst: int, carried_weight: float, D, Db) -> float:
    """
    Compute the cost of traveling from node `src` to node `dst`
    while carrying a given amount of gold.

    """
    # No movement → no cost
    if src == dst:
        return 0.0

    alpha = problem._alpha
    beta = problem._beta

    # Distance cost + load-dependent penalty
    return D[src][dst] + ((alpha * carried_weight) ** beta) * Db[src][dst]


def get_Cost(problem: Problem, chromosome: list[int], D, Db) -> float:
    """
    Evaluates the cost of a chromosome using a greedy unload strategy.
    """
    cost = 0.0
    weight = 0.0
    current = 0

    for nxt in chromosome:
        # Cost of going directly to the next city while carrying current gold
        direct = compute_segment_cost(problem, current, nxt, weight, D, Db)
        # Cost of returning to depot first, unloading, then going to next city
        via_0 = (
            compute_segment_cost(problem, current, 0, weight, D, Db)
            + compute_segment_cost(problem, 0, nxt, 0.0, D, Db)
        )

        # Greedy choice: unload only if it is cheaper than going directly
        if via_0 < direct:
            cost += via_0
            weight = 0.0    # Gold is unloaded at the depot
            current = 0
        else:
            cost += direct

        # Collect gold at the visited city
        weight += problem._graph.nodes[nxt].get("gold", 1)
        current = nxt
        
     # Final return to the depot with any remaining gold
    cost += compute_segment_cost(problem, current, 0, weight, D, Db)
    return cost
