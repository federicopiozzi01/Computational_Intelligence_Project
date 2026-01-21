from Problem import Problem
from src.cost import compute_segment_cost


def append_shortest_path_segment(
    output_path: list[tuple[int, float]],
    shortest_path: list[int],
    *,
    include_start_node: bool,
    pickup_at_end: float,
):
    """
    Append a shortest-path segment to the output path.

    - Intermediate nodes are appended as (node, 0.0)
    - The destination node is appended as (end, pickup_at_end)
    - If include_start_node=False, the first node of shortest_path is skipped (to avoid duplicates)
    """
    if not shortest_path:
        return

    start_idx = 0 if include_start_node else 1
    if start_idx >= len(shortest_path):
        return

    # --- intermediate nodes: pickup 0.0 ---
    for node in shortest_path[start_idx:-1]:
        tup = (node, 0.0)

        if not output_path:
            output_path.append(tup)
            continue

        last_node, last_pick = output_path[-1]
        if last_node == node:
            output_path[-1] = (last_node, max(last_pick, 0.0))
        else:
            output_path.append(tup)

    # --- destination node: pickup_at_end ---
    end_node = shortest_path[-1]
    end_pick = float(pickup_at_end)

    if not output_path:
        output_path.append((end_node, end_pick))
        return

    last_node, last_pick = output_path[-1]
    if last_node == end_node:
        output_path[-1] = (last_node, max(last_pick, end_pick))
    else:
        output_path.append((end_node, end_pick))


def build_output_path(problem: Problem, chromosome: list[int], D, Db, P):
    """
    Build the final path in the required format:

    - (0,0) at the beginning: IMPLICIT
    - (0,0) in the middle: MUST be included when unloading
    - (0,0) at the end
    - All nodes on the chosen shortest paths must appear in the output
      (with pickup = 0.0 when no gold is collected there)

    """
    # defensive: if chromosome is None or empty, just end at depot (0,0)
    if not chromosome:
        return [(0, 0.0)]

    out: list[tuple[int, float]] = []
    weight = 0.0
    current = 0

    for nxt in chromosome:
        direct = compute_segment_cost(problem, current, nxt, weight, D, Db)
        via_0 = (
            compute_segment_cost(problem, current, 0, weight, D, Db)
            + compute_segment_cost(problem, 0, nxt, 0.0, D, Db)
        )

        if via_0 < direct:
            # Go current -> 0 (with weight) and explicitly include depot (0,0)
            if current != 0:
                append_shortest_path_segment(
                    out,
                    P[current][0],
                    include_start_node=False,
                    pickup_at_end=0.0,
                )
            else:
                # already at 0: add (0,0) only if not already the last element
                if not out or out[-1][0] != 0:
                    out.append((0, 0.0))

            weight = 0.0
            current = 0

        # Go current -> nxt
        gold = float(problem._graph.nodes[nxt].get("gold", 1))

        # if we're at the start (implicit 0), do NOT print the initial 0
        include_start = not (current == 0 and len(out) == 0)

        append_shortest_path_segment(
            out,
            P[current][nxt],
            include_start_node=include_start,
            pickup_at_end=gold,
        )

        weight += gold
        current = nxt

    # Final return to depot: (0,0)
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

    # Safety: ensure last tuple is exactly (0,0)
    if not out or out[-1] != (0, 0.0):
        # avoid duplicate consecutive 0
        if not out or out[-1][0] != 0:
            out.append((0, 0.0))
        else:
            out[-1] = (0, 0.0)

    return out
