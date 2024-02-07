import random
example = [[(1,2),(3,4)],[(5,6),(7,8)],[(1,2),(3,4)],[],[]]

class SBM():
			'''
			Stochastic Block Model
			size (int)
			partition (list[list[int]])
			probabilities (list[list[int]])
			'''

			def __init__(self, size, partition, probabilities) -> None:
					assert isDisjoint(partition)
					adjList = [[] for _ in range(size)]
					for i in range(size):
							adjList[i].append(i)
							for j in range(size):
									if(i !=j):
											if(random.random() < probabilities[find_partition(partition, i)][find_partition(partition, j)]):
													adjList[i].append(j)
					print(adjList)
			

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
			
			
class ER(SBM):
			'''
			Erdős–Rényi model
			size (int)
			partition (list[list[int]])
			p (int)
			'''
			def __init__(self, size, partition, p) -> None:
						r = len(partition)
						probabilities = [[p for _ in range(r)] for _ in range(r)]
						super().__init__(size, partition, probabilities)
