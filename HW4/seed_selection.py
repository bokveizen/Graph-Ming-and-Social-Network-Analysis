# No external imports are allowed other than numpy
import numpy as np


def seed_selection(graph, policy, n):
    """
    Implement the function that chooses n initial active nodes.

    Inputs:
        graph: directed input graph in the edge list format.
            That is, graph is a list of tuples of the form (u, v),
            which indicates that there is an edge u to v.
            You can assume that both u and v are integers, while you cannot
            assume that the integers are within a specific range.
        policy: policy for selecting initial active nodes. ('degree', 'random', or 'custom')
            if policy == 'degree', n nodes with highest degrees are chosen
            if policy == 'random', n nodes are randomly chosen
            if policy == 'custom', n nodes are chosen based on your own policy
        n: number of initial active nodes you should choose

    Outputs: a list of n integers, corresponding to n nodes.
    """
    if policy == 'degree':
        degree_table = {}
        for u, v in graph:
            degree_table[u] = degree_table.get(u, 0) + 1
            degree_table[v] = degree_table.get(v, 0) + 1
        return sorted(degree_table.keys(), key=degree_table.get, reverse=True)[:n]

    elif policy == 'random':
        nodes = set()
        for e in graph:
            nodes.update(e)
        return list(np.random.choice(list(nodes), n, replace=False))

    elif policy == 'custom':
        in_deg_table = {}
        out_nbr_table = {}
        nodes = set()
        for u, v in graph:
            nodes.update((u, v))
            in_deg_table[v] = in_deg_table.get(v, 0) + 1
            out_nbr_table[u] = out_nbr_table.get(u, set()) | {v}
        scores = {}
        for u in out_nbr_table:
            scores[u] = sum(1/in_deg_table[v] for v in out_nbr_table[u])
        return sorted(scores.keys(), key=scores.get, reverse=True)[:n]

    else:
        raise NameError('Unsupported policy name')
