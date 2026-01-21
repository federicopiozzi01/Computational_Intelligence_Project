from Problem import Problem


def compute_segment_cost(problem: Problem, src: int, dst: int, carried_weight: float, D, Db) -> float:
    """
    Computes the edge-by-edge cost of traveling from src to dst
    with the given carried weight.
    """
    if src == dst:
        return 0.0

    alpha = problem._alpha
    beta = problem._beta
    return D[src][dst] + ((alpha * carried_weight) ** beta) * Db[src][dst]


def get_Cost(problem: Problem, chromosome: list[int], D, Db) -> float:
    """
    Evaluates the cost of a chromosome using a greedy unload strategy.
    """
    cost = 0.0
    weight = 0.0
    current = 0

    for nxt in chromosome:
        direct = compute_segment_cost(problem, current, nxt, weight, D, Db)
        via_0 = (
            compute_segment_cost(problem, current, 0, weight, D, Db)
            + compute_segment_cost(problem, 0, nxt, 0.0, D, Db)
        )

        if via_0 < direct:
            cost += via_0
            weight = 0.0
            current = 0
        else:
            cost += direct

        weight += problem._graph.nodes[nxt].get("gold", 1)
        current = nxt

    cost += compute_segment_cost(problem, current, 0, weight, D, Db)
    return cost
