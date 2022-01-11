#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 14:48:37 2021

@author: anpo
"""

# compute pairwise corr among columns

from pandas import *
import numpy as np
# from libraries.settings import *
from scipy.stats.stats import pearsonr
import itertools

cont_var_index = [21,22,23,29,41,42]
cont_var_index.extend(list(range(363,392)))

a = X_train_c[:,cont_var_index]

heads = [encoding_head[i] for i in cont_var_index]


df = DataFrame(a, columns=heads) 
print(df)


import numpy as np
import matplotlib.pyplot as plt

plt.imshow(df.corr('pearson'))
plt.colorbar()

# locs, labels = xticks()  # Get the current locations and labels.
plt.xticks( np.array(range(0,35)),heads)
plt.xticks(fontsize=5, rotation=90)

plt.yticks( np.array(range(0,35)),heads)
plt.yticks(fontsize=5)
# xticks(np.arange(0, 1, step=0.2))  # Set label locations.

# xticks(np.arange(3), ['Tom', 'Dick', 'Sue'])  # Set text labels.

# xticks(heads,

#        rotation=20)  # Set text labels and properties.

plt.savefig('f_corr.png', dpi=300)
plt.show()


# correlations = {}
# columns = df.columns.tolist()

# for col_a, col_b in itertools.combinations(columns, 2):
#     correlations[col_a + '__' + col_b] = pearsonr(df.loc[:, col_a], df.loc[:, col_b])

# result = DataFrame.from_dict(correlations, orient='index')
# result.columns = ['PCC', 'p-value']

# print(result.sort_index())

# 21 brtcnt

# 22 spo2
# 23 ddvisit30

# 29 ddvisit365

# 41
# 42
# 363
# 364
# 392