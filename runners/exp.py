import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from model.graph import *
from experiment import rct
from model import pom
from experiment.estimators import estimator
import seaborn as sns
import matplotlib.pyplot as plt

n_iters = [200 * i for i in range(7)]

nc = 50         # number of clusters
p_in = 0.5      # probability of edge between units within same cluster; largest this can be is (nc/n)*avg_degree
p_out = 0       # probability of edge between units of different clusters

### Other parameters
beta = 2        # potential outcomes model degree
p = 0.5         # treatment budget (marginal treatment probability)     
gr = 10          # number of graph repetitions
r = 100       # number of RCT repetitions
cf = lambda i, S, G: pom.uniform_coeffs(i, S, G)    # function to generate coefficients

### Experiment
