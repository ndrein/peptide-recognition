import numpy as np
from embed import get_distance_matrix
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from Bio.SubsMat.MatrixInfo import blosum62
from itertools import permutations, product
import pandas as pd


aas = sorted({t[0] for t in sorted(blosum62.keys())})

embedding = MDS(n_components=2, dissimilarity='precomputed').fit_transform(get_distance_matrix())
print(embedding)

fig, ax = plt.subplots()
ax.scatter(embedding[:, 0], embedding[:, 1])

for i, aa in enumerate(aas):
    ax.annotate(aa, (embedding[i, 0], embedding[i, 1]))
plt.show()
