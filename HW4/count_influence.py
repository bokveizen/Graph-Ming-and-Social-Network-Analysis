# No external imports are allowed other than numpy
import numpy as np


def count_influence(graph, seeds, threshold):
    """
    Implement the function that counts the ultimate number of active nodes.

    Inputs:
        graph: directed input graph in the edge list format.
            That is, graph is a list of tuples of the form (u, v),
            which indicates that there is an edge u to v.
            You can assume that both u and v are integers, while you cannot
            assume that the integers are within a specific range.
        seeds: a list of initial active nodes.
        threshold: the propagation threshold of the Independent Cascade Model.

    Output: the number of active nodes at time infinity.
    """
    in_deg_table = {}
    out_nbr_table = {}
    for u, v in graph:
        in_deg_table[v] = in_deg_table.get(v, 0) + 1
        out_nbr_table[u] = out_nbr_table.get(u, set()) | {v}
    active_nodes = set(seeds)
    last_activated = seeds
    while last_activated:
        influence_count = {}
        for u in last_activated:
            for v in out_nbr_table.get(u, set()):
                if v not in active_nodes:
                    influence_count[v] = influence_count.get(v, 0) + 1
        last_activated = []
        for v in influence_count:
            prob = 1 - (1 - threshold / in_deg_table[v]) ** influence_count[v]
            if np.random.random_sample() <= prob:
                last_activated.append(v)
                active_nodes.add(v)
    return len(active_nodes)


if __name__ == '__main__':
    from seed_selection import seed_selection

    with open('soc-Epinions.txt') as f:
        data = f.readlines()
    graph = [tuple(int(v) for v in line[:-1].split('\t')) for line in data]
    nodes = set()
    for e in graph:
        nodes.update(e)
    total_nodes_num = len(nodes)

    runs = [3]

    if 1 in runs:
        # 1.1, degree/random, q = 1.0, n = total_nodes_num * 0.1%/0.25%/0.5%/1%/2.5%/5%
        policies = ['degree', 'random']
        q = 1.0
        n_list = [total_nodes_num // r for r in [1000, 400, 200, 100, 40, 20]]
        random_seeds_num = 5
        simulation_times = 10
        for p in policies:
            repeat_times = random_seeds_num if p == 'random' else 1
            for i in range(repeat_times):
                for n in n_list:
                    seed = seed_selection(graph, p, n)
                    count_list = []
                    for t in range(simulation_times):
                        count_list.append(count_influence(graph, seed, q))
                    print('Seed using policy {} #{} with {} initial nodes: {} ± {}'
                          .format(p, i, n, np.mean(count_list), np.std(count_list).round(2)))

    if 2 in runs:
        # 1.2, degree/random, q = 0.2/0.4/0.6/0.8/1.0, n = total_nodes_num * 1%
        policies = ['degree', 'random']
        q_list = [0.2, 0.4, 0.6, 0.8, 1.0]
        n = total_nodes_num // 100
        random_seeds_num = 5
        simulation_times = 10
        for p in policies:
            repeat_times = random_seeds_num if p == 'random' else 1
            for i in range(repeat_times):
                seed = seed_selection(graph, p, n)
                for q in q_list:
                    count_list = []
                    for t in range(simulation_times):
                        count_list.append(count_influence(graph, seed, q))
                    print('Seed using policy {} #{} with propagation threshold {}: {} ± {}'
                          .format(p, i, q, np.mean(count_list), np.std(count_list).round(2)))

    if 3 in runs:
        # 2.1, custom, q = 1.0, n = total_nodes_num * 1%
        policies = ['custom']
        q = 1.0
        n = total_nodes_num // 100
        # random_seeds_num = 5
        simulation_times = 10
        for p in policies:
            seed = seed_selection(graph, p, n)
            count_list = []
            for t in range(simulation_times):
                count_list.append(count_influence(graph, seed, q))
            print('Seed using policy {} with propagation threshold {}: {} ± {}'
                  .format(p, q, np.mean(count_list), np.std(count_list).round(2)))
