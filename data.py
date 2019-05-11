from Bio.SeqIO import parse
import matplotlib.pyplot as plt
import numpy as np

d = np.array([len(r) for r in parse('uniprot_sprot.fasta', 'fasta')])
# print(d)

plt.hist(d[(d > 100) & (d < 400)])
plt.show()

print(len(d))
print(len(d[d > 500]))

print('median', np.median(d))

subset = d[(d > 100) & (d < 400)]
print(len(subset))
print('media', np.median(subset))
