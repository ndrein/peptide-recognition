import numpy as np
from itertools import product
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from Bio.SubsMat.MatrixInfo import blosum62
from itertools import permutations, product
import pandas as pd
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from Bio.SubsMat.MatrixInfo import blosum62
from itertools import permutations, product
import pandas as pd


def get_distance_matrix():
    aas = sorted({t[0] for t in sorted(blosum62.keys())})
    D = np.array([[blosum62[a, b] if (a, b) in blosum62 else blosum62[b, a] for a in aas] for b in aas])
    D = -(D + min(D.flatten()) + 1)
    return D


def embed(x_train, y_train, output_csv=None):
    max_len = max([len(s) for s in x_train])

    aas = sorted({t[0] for t in sorted(blosum62.keys())})
    D = get_distance_matrix()
    aa_to_embedding = {aa: embedding[0] for aa, embedding in
                       zip(aas, MDS(n_components=1, dissimilarity='precomputed').fit_transform(D))}

    processed = np.zeros((len(x_train), max_len))
    for i, s in enumerate(x_train):
        for j, c in enumerate(s):
            processed[i, j] = aa_to_embedding[c]

    if output_csv:
        print('Saving embedded csv')
        pd.DataFrame(np.c_[processed, y_train]).to_csv(output_csv, header=None, index=False, float_format='%.3f')
    return processed, y_train


if __name__ == '__main__':
    # x_train, y_train = np.hsplit(pd.read_csv('train.csv', delimiter=',', header=None).to_numpy(), 2)
    # x_train, y_train = x_train[:, 0], y_train[:, 0]
    # # x_train, y_train = x_train[:, 0], y_train[:, 0]
    # x_train = embed(x_train, y_train, 'embedded.csv')
    k = 3
    aas = sorted({t[0] for t in sorted(blosum62.keys())})
    kmer_to_index = {''.join(kmer): i + 1 for i, kmer in enumerate(product(aas, repeat=k))}
    print(kmer_to_index)
    print(len(kmer_to_index))
    # print(list(product(aas, repeat=3)))
    # product([1,2,3], repeat=3)
    # print([1, 2, 3, 4, 5][:-3 + 1])

    data = pd.read_csv('train.csv', delimiter=',', header=None).to_numpy()
    #
    x_train, y_train = data[:, 0], data[:, 1]
    print(x_train.shape, y_train.shape)

    max_len = max([len(s) for s in x_train])
    processed = np.zeros((len(x_train), max_len - k + 2))
    print(processed.shape)
    processed[:, -1] = y_train
    for i, s in enumerate(x_train):
        for j in range(len(s) - k + 1):
            processed[i, j] = kmer_to_index[s[j:j + k]] + 1

    print(processed.shape)

    # pd.DataFrame(processed).to_csv('kmer_embedded.csv', header=None, index=False)
    np.savetxt('kmer_embedded.csv', processed, delimiter=',')
