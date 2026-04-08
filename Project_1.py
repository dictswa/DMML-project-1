# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 10:58:33 2026

@author: Sabina
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("jain.txt", sep="\s+", header=None)

x = df[0]
y = df[1]

plt.scatter(x,y)
plt.show()
