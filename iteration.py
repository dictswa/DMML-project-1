# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 11:23:06 2026

@author: dicte
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import dbscan, KMeans, k_means, DBSCAN, HDBSCAN

df = pd.read_csv("jain.txt", sep="\s+", header=None)
'''
The goal scatterplot
'''
plt.scatter(df[0],df[1],c=df[2])
plt.show()

del df[2]

'''
K-means
'''
model = KMeans(n_clusters=2)
model.fit(df)
labels = model.predict(df)

plt.scatter(df[0],df[1],c=labels)
plt.show()

##### lloyd algorithm #####
model2 = k_means(df, n_clusters=2)
labels2 = model2[1]
plt.scatter(df[0],df[1],c=labels2)
plt.show()

##### elkan agorithm #####
model3 = k_means(df, n_clusters=2, algorithm = 'elkan')
labels3 = model3[1]
plt.scatter(df[0],df[1],c=labels3)
plt.show()

'''
DB-scan
'''
model4 = DBSCAN(eps=3,min_samples = 15).fit(df)
labels4 = model4.labels_
n_clusters_ = len(set(labels4)) - (1 if -1 in labels4 else 0)
n_noise_ = list(labels4).count(-1)

print("Estimated number of clusters: %d" % n_clusters_)
print("Estimated number of noise points: %d" % n_noise_)

plt.scatter(df[0],df[1],c=labels4)
plt.show()

model5 = HDBSCAN(min_cluster_size=10).fit(df)
labels5 = model5.labels_
n_clusters_ = len(set(labels5)) - (1 if -1 in labels5 else 0)
n_noise_ = list(labels5).count(-1)

print("Estimated number of clusters: %d" % n_clusters_)
print("Estimated number of noise points: %d" % n_noise_)

plt.scatter(df[0],df[1],c=labels5)
plt.show()


