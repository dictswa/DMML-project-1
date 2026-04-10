# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 10:48:24 2026

@author: Sabina
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

df = pd.read_csv("jain.txt", sep="\s+", header=None)

x = df[0]
y = df[1]
z = df[2]

model = KMeans(n_clusters=2)
model.fit(df)

labels = model.predict(df)

plt.scatter(x,y,c=labels)
plt.show()

score = silhouette_score(df, labels)
print(score)