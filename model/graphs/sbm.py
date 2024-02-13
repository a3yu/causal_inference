import random
from .graph import Graph

class SBM(Graph):
    '''
    Stochastic Block Model
    size (int)
    partition (list[list[int]])
    probabilities (list[list[int]])
    '''
    def __init__(self, size, partition, probabilities) -> None:
        assert isDisjoint(partition)
        self.adjList = [[] for _ in range(size)]
        for i in range(size):
            self.adjList[i].append(i)
            for j in range(size):
                if (i !=j):
                    if (random.random() < probabilities[find_partition(partition, i)][find_partition(partition, j)]):
                        self.adjList[i].append(j)

    def __str__(self) -> str:
        return str(self.adjList)
    

def isDisjoint(partition) -> bool:
        distinct = set()
        counter = 0
        for part in partition:
            for v in part:
                distinct.add(v)
                counter += 1
        return len(distinct) == counter

def find_partition(partition, node):
    for i in range(len(partition)):
        for j in range(len(partition[i])):
            if partition[i][j] == node:
                return i
    return -1
