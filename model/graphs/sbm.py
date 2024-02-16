import random
from .graph import Graph
import typing
class SBM(Graph):
    '''
    Stochastic Block Model
    size (int)
    partition (list[list[int]])
    probabilities (list[list[int]])
    '''
    def __init__(self, sizes, partitions, probabilities) -> None:
        # exactly one of the three parameters are lists of size > 1, 
        # the other 2 are singleton lists
        self.sizes = sizes
        self.partitions = partitions
        self.probabilities = probabilities

    def generateGraphs(self):
        results = []
        if len(self.sizes) > 1:
            partition = self.partitions[0]
            P = self.probabilities[0]
            for size in self.sizes:
                graph = self._generate_graph(size, partition, P)
                results.append(graph)
        
        elif len(self.partitions) > 1:
            size = self.sizes[0]
            P = self.probabilities[0]
            for partition in self.partitions:
                graph = self._generate_graph(size, partition, P)
                results.append(graph)

        elif len(self.probabilities) > 1:
            size = self.sizes[0]
            partition = self.partitions[0]
            for P in self.probabilities:
                graph = self._generate_graph(size, partition, P)
                results.append(graph)
        
        return results


    def _generate_graph(self, size, partition, P):
        adjList = [[] for _ in range(size)]
        for i in range(size):
            adjList[i].append(i)
            for j in range(size):
                if i != j:
                    if (random.random() < P[find_partition(partition, i)][find_partition(partition, j)]):
                        adjList[i].append(j)
        
        return adjList

    def __str__(self) -> str:
        return str(self.adjList)
    

# def isDisjoint(partition) -> bool:
#         distinct = set()
#         counter = 0
#         for part in partition:
#             for v in part:
#                 distinct.add(v)
#                 counter += 1
#         return len(distinct) == counter

def find_partition(partition, node):
    for i in range(len(partition)):
        for j in range(len(partition[i])):
            if partition[i][j] == node:
                return i
    return -1
