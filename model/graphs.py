'''
Implementation of basic graphs (without using classes).
'''
import random
import numpy as np

def balanced_partition_pin_pout(n : int, nc : int, p_in : float, p_out: float):
    '''
    Returns partitions (list of "equal-sized" partitions of [n]) and probs (nc by nc numpy array with p_in on diagonal and p_out on offdiagonal)
    
    n : number of units
    nc: number of partitions/clusters/groups
    p_in: edge probability within clusters
    p_out: edge probability across clusters

    e.g. if n = 10 & nc = 2, partitions = [[0,1,2,3,4], [5,6,7,8,9]]
    if n = 10 & nc = 3, partitions = [[0,1,2,3], [4,5,6], [7,8,9]]
    if n = 10 & nc = 4, partitions = [[0,1,2], [3,4,5], [6,7], [8,9]]
    if n = 10 & nc = 5, partitions = [[0,1,2,3,4], [5,6,7,8,9]]
    '''
    probs = np.full((nc, nc), p_out, dtype=float)
    np.fill_diagonal(probs, p_in)

    if n % nc == 0: # we can split units evenly into nc clusters
        partition_size = n // nc
        partitions = [list(range(i * partition_size, (i + 1) * partition_size)) for i in range(nc)]
    else: # we cannot split into equal-sized; (n % nc) clusters will have (n//nc)+1 units; the remaining clusters will have (n//nc) units
        larger_partitions = n % nc
        smaller_partitions = nc - larger_partitions
        larger_partition_size = n // nc + 1
        smaller_partition_size = n // nc

        partitions = [list(range(i * larger_partition_size, (i + 1) * larger_partition_size)) for i in range(larger_partitions)]
        partitions += [list(range(larger_partitions * larger_partition_size + (i * smaller_partition_size),
                                    larger_partitions * larger_partition_size + ((i+1) * smaller_partition_size))) for i in range(smaller_partitions)]
    return partitions, probs


def SBM(size, partition, probabilities):
    '''
    Stochastic Block Model
    size (int): number of nodes in the graph
    partition (list[list[int]]): 2d list encoding the grouping of the nodes: must be disjoint
    probabilities (list[list[float]]): 2d list encoding probabilities for edges b/w each partition
    Returns: adjlist (list[list[int]])
    '''
    assert _isDisjoint(partition)
    # Precompute node to partition mapping
    node_to_partition = {}
    for partition_index, nodes in enumerate(partition):
        for node in nodes:
            node_to_partition[node] = partition_index

    adjList = [[] for _ in range(size)]
    
    for i in range(size):
        adjList[i].append(i)  # Add self-loop
        for j in range(size):
            if i != j:
                partition_i = node_to_partition[i]
                partition_j = node_to_partition[j]
                if random.random() < probabilities[partition_i][partition_j]:
                    adjList[j].append(i)  # Add directed edge from i to j
    
    return adjList


def ER(size, partition, p):
    '''
    Erdős–Rényi model
    size (int): number of nodes in the graph
    partition (list[list[int]]): 2d list encoding the grouping of the nodes: must be disjoint
    p (float): probability of including an edge
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
