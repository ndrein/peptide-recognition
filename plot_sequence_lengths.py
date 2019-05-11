from Bio.SeqIO import parse
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from collections import Counter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

INPUT_FASTA = 'uniprot_sprot.fasta'
MIN_LENGTH = 100
MAX_LENGTH = 400
# OUT_FILE = 'train.csv'

np.random.seed(42)

print('Reading real sequences from', INPUT_FASTA)
real = np.array([str(r.seq) for r in parse(INPUT_FASTA, 'fasta')], dtype=object)
print(max([len(s) for s in real]))
print(min([len(s) for s in real]))
print(np.average([len(s) for s in real]))
plt.hist([len(s) for s in real], bins=1000)
plt.xlabel('Sequence Length')
plt.ylabel('# of Sequences')
plt.title('Swiss-Prot Length Distribution')
plt.show()


real = np.array([str(r.seq) for r in parse(INPUT_FASTA, 'fasta') if MIN_LENGTH < len(r) < MAX_LENGTH and 'U' not in r],
                dtype=object)
print(len(real))
us = [r for r in parse(INPUT_FASTA, 'fasta') if 'U' in r]
print('num U"s', len(us))
print(max([len(s) for s in real]))
print(min([len(s) for s in real]))
print(np.average([len(s) for s in real]))
plt.hist([len(s) for s in real])
plt.xlabel('Sequence Length')
plt.ylabel('# of Sequences')
plt.title('Filtered Sequences Length Distribution')
plt.show()


# print('AA frequencies')
# aa_to_freq = Counter()
# for s in real:
#     aa_to_freq.update(s)
# num_aas = sum(aa_to_freq.values())
# aa_to_freq = {aa: aa_to_freq[aa] / num_aas for aa in aa_to_freq}
#
# print('Generating fake chain lengths')
# lengths = [int(length) for length in gaussian_kde([len(s) for s in real]).resample(len(real))[0] if length > 0]
# print('Generating fake peptides')
# fake = np.array([''.join(np.random.choice(a=list(aa_to_freq.keys()), size=length, p=list(aa_to_freq.values())))
#                  for length in lengths])
#
# print('Concatenating real/fake peptides')
# data = np.r_[np.c_[real, np.zeros(len(real), dtype=int)], np.c_[fake, np.ones(len(fake), dtype=int)]]
#
# print('Writing csv')
# pd.DataFrame(data).sample(frac=1).to_csv(OUT_FILE, header=None, index=False)
