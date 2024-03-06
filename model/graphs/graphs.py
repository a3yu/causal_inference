import random

def SBM(size, partition, probabilities):
    '''
    Stochastic Block Model
    size (int): number of nodes in the graph
    partition (list[list[int]]): 2d list encoding the grouping of the nodes: must be disjoint
    probabilities (list[list[int]]): 2d list encoding probabilities for edges b/w each partition
    Returns: adjlist (list[list[int]])
    '''
    assert _isDisjoint(partition)
    def find_partition(partition, node):
        for i in range(len(partition)):
            for j in range(len(partition[i])):
                if partition[i][j] == node:
                    return i
        return -1
    
    adjlist = [[] for _ in range(size)]
    for i in range(size):
        adjlist[i].append(i)
        for j in range(size):
            if i != j and (random.random() < probabilities[find_partition(partition, i)]
                            [find_partition(partition, j)]):
                adjlist[i].append(j)
        return adjlist


def ER(size, partition, p):
    '''
    Erdős–Rényi model
    size (int): number of nodes in the graph
    partition (list[list[int]]): 2d list encoding the grouping of the nodes: must be disjoint
    p (int): probability of including an edge
    Returns: adjlist (list[list[int]])
    '''
    assert _isDisjoint(partition)
    r = len(partition)
    probabilities = [[p for _ in range(r)] for _ in range(r)]
    return SBM(size, partition, probabilities)


def _isDisjoint(partition) -> bool:
        distinct = set()
        counter = 0
        for part in partition:
            for v in part:
                distinct.add(v)
                counter += 1
        return len(distinct) == counter