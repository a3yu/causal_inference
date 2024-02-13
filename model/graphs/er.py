from sbm import SBM
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