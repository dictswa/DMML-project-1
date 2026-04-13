# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 11:23:06 2026

@author: dicte
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, k_means, DBSCAN, HDBSCAN, OPTICS, AgglomerativeClustering
from sklearn.mixture import GaussianMixture

df = pd.read_csv("jain.txt", sep="\s+", header=None)
'''
The goal scatterplot
'''
plt.scatter(df[0],df[1],c=df[2])
plt.title("Jain dataset")
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
model4 = DBSCAN(eps=2.5,min_samples = 3).fit(df)
labels4 = model4.labels_
n_clusters_ = len(set(labels4)) - (1 if -1 in labels4 else 0)
n_noise_ = list(labels4).count(-1)

print("Estimated number of clusters: %d" % n_clusters_)
print("Estimated number of noise points: %d" % n_noise_)

plt.scatter(df[0],df[1],c=labels4)
plt.show()

model5 = HDBSCAN(min_cluster_size=6).fit(df)
labels5 = model5.labels_
n_clusters_ = len(set(labels5)) - (1 if -1 in labels5 else 0)
n_noise_ = list(labels5).count(-1)

print("Estimated number of clusters: %d" % n_clusters_)
print("Estimated number of noise points: %d" % n_noise_)

plt.scatter(df[0],df[1],c=labels5)
plt.show()

'''
Optics
'''
model6 = OPTICS(min_samples=15, metric="cosine").fit(df)
labels6 = model6.labels_
plt.scatter(df[0],df[1],c=labels6)
plt.show()


model7 = AgglomerativeClustering(linkage = 'ward').fit(df)
labels7 = model7.labels_
plt.subplot(2,3,1)
plt.scatter(df[0],df[1],c=labels7, s=1)
plt.axis('off')
plt.title("Agglo Ward")

model8 = AgglomerativeClustering(linkage = 'single').fit(df)
labels8 = model8.labels_
plt.subplot(2,3,2)
plt.scatter(df[0],df[1],c=labels8, s=1)
plt.axis('off')
plt.title("Agglo Single")

model9 = AgglomerativeClustering(linkage = 'complete').fit(df)
labels9 = model9.labels_
plt.subplot(2,3,4)
plt.scatter(df[0],df[1],c=labels9, s=1)
plt.axis('off')
plt.title("Agglo Complete")

model10 = AgglomerativeClustering(linkage = 'average').fit(df)
labels10 = model10.labels_
plt.subplot(2,3,5)
plt.scatter(df[0],df[1],c=labels10, s=1)
plt.axis('off')
plt.title("Agglo Average")

plt.subplot(2,3,6)
plt.scatter(df[0],df[1],c=labels4, s=1)
plt.axis('off')
plt.title("DB-scan")

plt.subplot(2,3,3)
plt.scatter(df[0],df[1],c=labels, s=1)
plt.axis('off')
plt.title("K-means")

plt.show()

model11 = GaussianMixture(n_components = 2,init_params='random', random_state=0)
labels11 = model11.fit_predict(df)
plt.scatter(df[0],df[1],c=labels11)
plt.show()

