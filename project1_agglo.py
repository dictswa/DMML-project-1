# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 12:03:09 2026

@author: Sabina
"""

import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

df = pd.read_csv("jain.txt", sep="\s+", header=None)
del df[2]
x = df[0]
y = df[1]

model = AgglomerativeClustering(n_clusters=2, linkage='ward')
labels = model.fit_predict(df)

plt.scatter(x,y,c=labels)
plt.show()

score = silhouette_score(df, labels)
print(score)