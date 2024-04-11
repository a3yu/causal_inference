'''
Testing modules for graphs using unittest
'''
import random
from model.graphs import SBM
from model.graph import SBM as SBMGraph
from model.graph import ER
from model.graph import SimpleSBM
from collections import Counter
import unittest

import random
import unittest

class TestSimpleSBM(unittest.TestCase):
    
    def setUp(self):
        random.seed(42)

    def test_graph_initialization(self):
        size = 10
        partitionAmount = 2
        inside = 0.8
        outside = 0.1
        graph = SimpleSBM(size, partitionAmount, inside, outside)

        # Test basic properties
        self.assertEqual(len(graph), size)
        for node in range(size):
            self.assertIn(node, graph[node], "Missing self-loop at node {}".format(node))

    def test_edge_distribution(self):
        size = 100  # Larger graph for statistical significance
        partitionAmount = 5
        inside = 0.5
        outside = 0.05
        graph = SimpleSBM(size, partitionAmount, inside, outside)

        node_to_partition = {}
        partition_size = size // partitionAmount
        for p in range(partitionAmount):
            for node in range(p * partition_size, min((p + 1) * partition_size, size)):
                node_to_partition[node] = p

        # Verify partition sizes
        partition_counts = Counter(node_to_partition.values())
        for count in partition_counts.values():
            self.assertTrue(count == partition_size or count == partition_size + 1)

        edge_counts = Counter()
        for i in range(size):
            for j in graph[i]:
                if i != j:  # Exclude self-loops
                    i_part = node_to_partition[i]
                    j_part = node_to_partition[j]
                    edge_counts[(i_part, j_part)] += 1

        # Check if the number of edges is consistent with the probabilities
        for (i_part, j_part), count in edge_counts.items():
            if i_part == j_part:
        # For intra-partition edges, the possible edge count is the combination of nodes in the partition
                possible_edges = partition_counts[i_part] * (partition_counts[i_part] - 1) / 2
                expected_count = inside * possible_edges
            else:
        # For inter-partition edges, the possible edge count is the product of the sizes of the two partitions
                possible_edges = partition_counts[i_part] * partition_counts[j_part]
                expected_count = outside * possible_edges

    # Check if the actual number of edges is close to the expected number
        self.assertAlmostEqual(count, expected_count, delta=0.1 * possible_edges)


    def test_no_connection_between_partitions(self):
        size = 30
        partitionAmount = 3
        inside = 1.0  # Ensure complete connectivity within partitions
        outside = 0.0  # Ensure no connectivity between partitions
        graph = SimpleSBM(size, partitionAmount, inside, outside)

        partition_size = size // partitionAmount
        for i in range(partitionAmount):
            start = i * partition_size
            end = start + partition_size
            partition_nodes = set(range(start, end))

            for node in range(start, end):
                self.assertTrue(set(graph[node]) - {node} == partition_nodes - {node},
                                f"Node {node} in partition {i} is not correctly connected within its partition.")

    def test_randomized_edge_creation(self):
        size = 50
        partitionAmount = 5
        inside = 0.3
        outside = 0.1
        graph = SimpleSBM(size, partitionAmount, inside, outside)

        # Check that the edge probabilities fall within a reasonable range
        inside_edges = 0
        outside_edges = 0
        total_inside_possible = 0
        total_outside_possible = 0
        partition_size = size // partitionAmount

        for i in range(size):
            i_part = i // partition_size
            for j in graph[i]:
                if i != j:
                    j_part = j // partition_size
                    if i_part == j_part:
                        inside_edges += 1
                    else:
                        outside_edges += 1

            total_inside_possible += partition_size - 1  # Exclude self-loop
            total_outside_possible += size - partition_size

        self.assertAlmostEqual(inside_edges / total_inside_possible, inside, delta=0.1)
        self.assertAlmostEqual(outside_edges / total_outside_possible, outside, delta=0.1)

class TestSBM(unittest.TestCase):

    def setUp(self):
        """Set up a predictable random seed for consistent test results."""
        random.seed(42)

    def test_graph_size_and_self_loops(self):
        """Test that the SBM generates a graph of the correct size and includes self-loops for each node."""
        size = 5
        partition = [[0, 1], [2, 3], [4]]
        probabilities = [[0.1, 0.2, 0.3], [0.2, 0.1, 0.4], [0.3, 0.4, 0.1]]
        graph = SBM(size, partition, probabilities)
        self.assertEqual(len(graph), size, "Graph size does not match expected.")
        for i, edges in enumerate(graph):
            self.assertIn(i, edges, "Missing self-loop at node {}".format(i))

    def test_edge_creation_probability(self):
        """Ensure that the edges are created according to the specified probabilities."""
        size = 4
        partition = [[0, 1], [2, 3]]
        # Set probabilities such that nodes within the same partition won't connect, but will always connect to the other partition
        probabilities = [[0.0, 1.0], [1.0, 0.0]]
        graph = SBM(size, partition, probabilities)

        # Check for the existence of inter-partition edges and absence of intra-partition edges
        self.assertNotIn(1, graph[0], "Intra-partition edge incorrectly exists.")
        self.assertIn(2, graph[1], "Expected inter-partition edge missing.")
        self.assertIn(3, graph[0], "Expected inter-partition edge missing.")
        self.assertNotIn(3, graph[2], "Intra-partition edge incorrectly exists.")

    def test_complete_connectivity_within_partition(self):
        """Test for complete connectivity within a partition when probability is 1."""
        size = 3
        partition = [[0, 1, 2]]  # All nodes in one partition
        probabilities = [[1.0]]  # Guarantee edge creation within the partition
        graph = SBM(size, partition, probabilities)

        # Since within-partition probability is 1, all nodes should be connected, including self-loops
        for i in range(size):
            for j in range(size):
                self.assertIn(j, graph[i], "Node {} should be connected to node {}".format(i, j))

    def test_edge_distribution_within_partition(self):
        size = 100  # Total number of nodes
        partition = [list(range(50)), list(range(50, 100))]  # Two partitions of 50 nodes each
        probabilities = [[0.1, 0.05], [0.05, 0.1]]  # Probability matrix
        
        graph = SBM(size, partition, probabilities)  # Generate the graph
        
        # Count the number of edges within each partition
        edge_counts = Counter()
        for i, edges in enumerate(graph):
            for j in edges:
                if i != j:  # Exclude self-loops
                    i_part = 0 if i < 50 else 1
                    j_part = 0 if j < 50 else 1
                    edge_counts[(i_part, j_part)] += 1
        
        # Calculate expected and observed edges
        for (from_part, to_part), count in edge_counts.items():
            # For intra-partition edges, excluding self-loops
            if from_part == to_part:
                size_part = 50  # Nodes in one partition
                probability = probabilities[from_part][to_part]
                expected_edges = probability * size_part * (size_part - 1)
                
                # Allow for some variance
                self.assertAlmostEqual(count, expected_edges, delta=expected_edges * 0.1,
                                       msg=f"Observed edges within partition {from_part} do not match expected.")
                
    def test_large_graph_edge_distribution(self):
        size = 1000  # Total number of nodes
        # Partition the graph into four equal parts
        partition = [list(range(250)), list(range(250, 500)), list(range(500, 750)), list(range(750, 1000))]
        # Set probabilities for edge creation between partitions
        # Higher probability within partitions, lower probability between different partitions
        probabilities = [
            [0.1, 0.05, 0.02, 0.05],
            [0.05, 0.1, 0.05, 0.02],
            [0.02, 0.05, 0.1, 0.05],
            [0.05, 0.02, 0.05, 0.1]
        ]
        
        graph = SBM(size, partition, probabilities)  # Generate the graph

        # Count the number of directed edges between each pair of partitions
        edge_counts = Counter()
        for i, edges in enumerate(graph):
            i_part = i // 250  # Determine the partition of node i
            for j in edges:
                if i != j:  # Exclude self-loops
                    j_part = j // 250  # Determine the partition of node j
                    edge_counts[(i_part, j_part)] += 1
        
        # Check if the observed number of edges matches the expected number
        for i in range(4):
            for j in range(4):
                size_i = len(partition[i])
                size_j = len(partition[j]) if i != j else size_i - 1  # Adjust for self-loops
                expected_edges = probabilities[i][j] * size_i * size_j
                observed_edges = edge_counts[(i, j)]
                
                # Allow a 10% variance due to the stochastic nature of the SBM
                self.assertAlmostEqual(observed_edges, expected_edges, delta=expected_edges * 0.1,
                                       msg=f"Observed edges from partition {i} to {j} do not match expected.")

class TestGraphs(unittest.TestCase):

    def setUp(self):
        random.seed(42)

    def testSBM(self):
        size = 5
        partition = [[0, 1], [2, 3], [4]]
        probabilities = [[0.1, 0.2, 0.3], [0.2, 0.1, 0.4], [0.3, 0.4, 0.1]]
        graph = SBMGraph(size, partition, probabilities)
        self.assertEqual(len(graph), size, "Graph size does not match expected.")
        for i, edges in enumerate(graph):
            self.assertIn(i, edges, "Missing self-loop at node {}".format(i))

    def testER(self):
        size = 10
        p = 0.5
        partition = [list(range(size))]  # Single partition for ER graph
        graph = ER(size, partition, p)
        
        self.assertEqual(graph.size, size, "Graph size incorrect after initialization.")
        self.assertEqual(len(graph.adjList), size, "Adjacency list size incorrect after initialization.")

if __name__ == '__main__':
    unittest.main()