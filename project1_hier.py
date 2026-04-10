# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 11:22:16 2026

@author: Sabina
"""

import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.metrics import silhouette_score

df = pd.read_csv("jain.txt", sep="\s+", header=None)
del df[2]

mergings = linkage(df, method='ward')
dendrogram(mergings,
           leaf_rotation=90,
           leaf_font_size=6)
plt.show()

