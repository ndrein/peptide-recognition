import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from Bio.SubsMat.MatrixInfo import blosum62
from itertools import permutations, product
import pandas as pd


# print(blosum62)
# print(len(blosum62))
def get_distance_matrix():
    aas = sorted({t[0] for t in sorted(blosum62.keys())})
    D = np.array([[blosum62[a, b] if (a, b) in blosum62 else blosum62[b, a] for a in aas] for b in aas])
    D = -(D + min(D.flatten()) + 1)
    return D
