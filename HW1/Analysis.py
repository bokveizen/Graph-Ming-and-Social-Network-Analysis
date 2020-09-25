import matplotlib.pyplot as plt
import numpy as np
import random
from collections import defaultdict, Counter

class Analysis:
    def __init__(self, adj, fileName) -> None:
        ############## #TODO: Complete the function ##################
        # Store adjacency matrix (estimated)
        ########################################################################
        self.fileName = fileName
        self.degrees = defaultdict(int)
        for row in adj:
            self.degrees[row.count_nonzero()] += 1
        self.plotDegDist()

    ######################### Implementation end ###########################

    ########################################################################
    # You may add additional functions for convenience                     #
    ########################################################################

    ######################### Implementation end ###########################

    def plotDegDist(self):
        plt.scatter(self.degrees.keys(), self.degrees.values())
        plt.xlabel('Degree')
        plt.ylabel('Count')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlim([max(min(self.degrees.keys()) - 10, 0.5), max(self.degrees.keys()) + 100])
        plt.savefig(self.fileName)

class Analysis_faster:
    def __init__(self, degrees, k, fileName) -> None:
        ############## #TODO: Complete the function ##################
        # Store adjacency matrix (estimated)
        ########################################################################
        self.fileName = fileName
        cur = degrees[:]
        for i in range(1, k):
            cur = np.kron(cur, degrees)
        # for d in cur:
        #     self.degrees[d] += 1
        self.degrees = Counter(cur)
        self.plotDegDist()

    ######################### Implementation end ###########################

    ########################################################################
    # You may add additional functions for convenience                     #
    ########################################################################

    ######################### Implementation end ###########################

    def plotDegDist(self):
        plt.scatter(self.degrees.keys(), self.degrees.values())
        plt.xlabel('Degree')
        plt.ylabel('Count')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlim([max(min(self.degrees.keys()) - 10, 0.5), max(self.degrees.keys()) + 100])
        plt.savefig(self.fileName)