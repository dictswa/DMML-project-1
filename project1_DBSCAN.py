# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 10:57:47 2026

@author: Sabina
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

df = pd.read_csv("jain.txt", sep="\s+", header=None)

x = df[0]
y = df[1]
z = df[2]

model = DBSCAN(eps=3, min_samples=10)

labels = model.fit_predict(df)

plt.scatter(x,y,c=labels)
plt.show()

score = silhouette_score(X, labels)
print(score)