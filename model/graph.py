import random

class Graph:
    def __init__(self, size) -> None:
        self.size = size
        self._generate_graph()

    
    def _generate_graph(self):
        self.adjList = [0 for _ in range(self.size)]

    
    def __getitem__(self, key):
        return self.adjList[key]
    

    def __setitem__(self, key, value):
        self.adjList[key] = value

    def __len__(self):
        return len(self.adjList)

class SBM(Graph):
    def __init__(self, size, partition, probabilities) -> None:
        self.partition = partition
        self.probabilities = probabilities
        super().__init__(size)

    def _generate_graph(self):
        '''
        Stochastic Block Model
        size (int): number of nodes in the graph
        partition (list[list[int]]): 2d list encoding the grouping of the nodes: must be disjoint
        probabilities (list[list[float]]): 2d list encoding probabilities for edges b/w each partition
        Returns: adjlist (list[list[int]])
        '''
        # precompute node-to-partition mapping
        node_to_partition = {}
        for partition_index, nodes in enumerate(self.partition):
            for node in nodes:
                node_to_partition[node] = partition_index

        adjList = [[] for _ in range(self.size)]
        
        for i in range(self.size):
            adjList[i].append(i)  # add self loop
            for j in range(self.size):
                if i != j:
                    partition_i = node_to_partition[i]
                    partition_j = node_to_partition[j]
                    if random.random() < self.probabilities[partition_i][partition_j]:
                        adjList[j].append(i)  # add directed edge from i to j
        
        self.adjList = adjList
class SimpleSBM(Graph):
    def __init__(self, size, partitionAmount, inside, outside) -> None:
        self.partitionAmount = partitionAmount
        self.inside = inside
        self.outside = outside
        self.partitions = [[] for _ in range(partitionAmount)]
        super().__init__(size)

    def _generate_graph(self):
        # Divide nodes into roughly equal partitions and create node-to-partition mapping
        node_to_partition = {}
        partition_size = self.size // self.partitionAmount
        
        for p in range(self.partitionAmount):
            start = p * partition_size
            end = min((p + 1) * partition_size, self.size)
            self.partitions[p] = list(range(start, end))
            for node in self.partitions[p]:
                node_to_partition[node] = p

        # Handle any remaining nodes in case of uneven division
        remaining_start = self.partitionAmount * partition_size
        if remaining_start < self.size:
            self.partitions[-1].extend(range(remaining_start, self.size))
            for node in range(remaining_start, self.size):
                node_to_partition[node] = self.partitionAmount - 1

        # Initialize adjacency list and add edges based on probabilities
        self.adjList = [[] for _ in range(self.size)]

        for i in range(self.size):
            self.adjList[i].append(i)  # add self loop
            for j in range(self.size):
                if i != j:
                    partition_i = node_to_partition[i]
                    partition_j = node_to_partition[j]
                    probability = self.inside if partition_i == partition_j else self.outside
                    if random.random() < probability:
                        self.adjList[i].append(j)  # add directed edge from i to j
class ErdosRenyi(Graph):
    def __init__(self, size, p) -> None:
        super().__init__(size)
        self.p = p
        self._generate_graph()
    
    def _generate_graph(self):
        for i in range(self.size):
            for j in range(i + 1, self.size):
                self.adjList[i].append(i)
                if random.random() < self.p:
                    self.adjList[i].append(j)