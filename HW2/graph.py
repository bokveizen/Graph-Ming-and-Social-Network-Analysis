"""This file contains an incomplete implementation of the CitationNetwork class and HITS algorithm.
Your tasks are as follows:
    1. Complete the CitationNetwork class
    2. Complete the hits method
    3. Complete the print_top_k method
"""

from __future__ import absolute_import
from typing import Dict, Tuple

############################################################################
# You may import additional python standard libraries, numpy and scipy.
# Other libraries are not allowed.
############################################################################
from scipy.sparse import coo_matrix
import numpy as np
from numpy import linalg as LA


class CitationNetwork:
    """Graph structure for the analysis of the citation network
    """

    def __init__(self, file_path: str) -> None:
        """The constructor of the CitationNetwork class.
        It parses the input file and generates a graph.

        Args:
            file_path (str): The path of the input file which contains papers and citations
        """
        ######### Task 1. Complete the constructor of CitationNetwork ##########
        # Load the input file and process it to a graph
        # You may declare any class variable or method if needed
        ########################################################################
        with open(file_path, encoding='utf-8') as f:
            d = f.readlines()
        self.total_num = int(d[0][:-1])
        row = []
        col = []
        cur_index = -1
        for i in range(1, len(d)):
            S = d[i]
            if S[:2] == '#i':
                cur_index = int(S[6:-1])
            elif S[:2] == '#%':
                ref_index = int(S[2:-1])
                row.append(cur_index)
                col.append(ref_index)
        data = [True for _ in row]
        self.G = coo_matrix((data, (row, col)), shape=(self.total_num, self.total_num))

    ############################################################################
    # You may add additional functions for convenience                         #
    ############################################################################


def hits(
        graph: CitationNetwork, max_iter: int, tol: float
) -> Tuple[Dict[int, float], Dict[int, float]]:
    """An implementation of HITS algorithm.
    It uses the power iteration method to compute hub and authority scores.
    It returns the hub and authority scores of each node.

    Args:
        graph (CitationNetwork): A CitationNetwork
        max_iter (int): Maximum number of iterations in the power iteration method
        tol (float): Error tolerance to check convergence in the power iteration method

    Returns:
        (hubs, authorities) (Tuple[Dict[int, float], Dict[int, float]]): Two-tuple of dictionaries.
            For each dictionary, the key is the paper index (int) and the value is its score (float)
    """
    ################# Task2. Complete the hits function ########################
    # Compute hub and authority scores of each node using the power iteration method
    ############################################################################
    total_num = graph.total_num
    H = np.array([1 / np.sqrt(total_num) for _ in range(total_num)])
    A = np.array([1 / np.sqrt(total_num) for _ in range(total_num)])
    for i in range(max_iter):
        # old_A = A.copy()
        old_H = H.copy()
        A = graph.G.transpose() * H
        H = graph.G * A
        A /= LA.norm(A)
        H /= LA.norm(H)
        if LA.norm(H - old_H) < tol:
            break
    H_dict = dict(zip(range(total_num), H))
    A_dict = dict(zip(range(total_num), A))
    return H_dict, A_dict


def print_top_k(scores: Dict[int, float], k: int) -> None:
    """Print top-k scores in the decreasing order and the corresponding indices.
    The printing format should be as follows:
        <Index 1>\t<score>
        <Index 2>\t<score>
        ...
        <Index k>\t<score>

    Args:
        scores (Dict[int, float]): Hub or Authority scores.
            For each dictionary, the key is the paper index (int) and the value is its score (float)
        k (int): The number of top scores to print.
    """

    ############## Task3. Complete the print_top_k function ####################
    # Print top-k scores in the decreasing order
    ############################################################################
    topk_ind = sorted(scores, key=scores.get, reverse=True)[:k]
    for ind in topk_ind:
        print('{}\t{}'.format(ind, scores[ind]))
