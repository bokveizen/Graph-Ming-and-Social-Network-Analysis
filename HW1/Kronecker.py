import numpy as np
from scipy.sparse import lil_matrix
import time

class KProds:
    def __init__(self, k: int, filePath: str) -> None:
        self.k = k  # The number of Kronecker products
        ############## #TODO: Complete the function ##################
        # Load the inital matrix from the file
        # You may declare any class variables if needed                        #
        ########################################################################
        with open(filePath) as f:
            a = f.readlines()
        self.P = np.array([[int(c) for c in line if c.isdigit()] for line in a])
        self.n = len(a)
        self.degrees = [sum(r) for r in self.P]
        ######################### Implementation end ###########################

    def produceGraph(self) -> lil_matrix:
        ############## #TODO: Complete the function ##################
        from scipy.sparse import hstack, vstack
        # Compute the k-th Kronecker power of the inital matrix
        ########################################################################
        cur = lil_matrix(self.P)
        for t in range(1, self.k):
            size = np.power(self.n, t)
            empty = lil_matrix((size, size), dtype=self.P.dtype)
            new = None
            for r in range(self.n):  # r-th row
                row = None
                for d in self.P[r]:
                    if d:
                        if row is None:
                            row = cur.copy()
                        else:
                            row = hstack([row, cur], format='lil')
                    else:
                        if row is None:
                            row = empty.copy()
                        else:
                            row = hstack([row, empty], format='lil')
                if new is None:
                    new = row.copy()
                else:
                    new = vstack([new, row], format='lil')
            cur = new.copy()
        return cur
        ######################### Implementation end ###########################


########################################################################
# You may add additional functions for convenience                        #
########################################################################
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    for i in range(1, 8):
        kprods = KProds(k=i, filePath='inputA.txt')
        adj = kprods.produceGraph()
        print(adj.toarray())
        plt.matshow(adj.toarray())
        plt.show()
    for i in range(1, 6):
        kprods = KProds(k=i, filePath='inputB.txt')
        adj = kprods.produceGraph()
        print(adj.toarray())
        plt.matshow(adj.toarray())
        plt.show()
######################### Implementation end ###########################
