'''
Density Estimation
'''
##### Using fixed epsilon value #####
dataset = {"A":(1,1), "B":(2,1), "C":(1,2), "D":(2,2), 
           "E":(3,5), "F":(3,9), "G":(3,10), "H":(4,10), 
           "I":(4,11), "J":(5,10), "K":(7,10), "L":(10,9), 
           "M":(10,6), "N":(9,5), "O":(10,5), "P":(11,5), 
           "Q":(9,4), "R":(10,4), "S":(11,4), "T":(10,3)}

def DE_eps(dataset, point, eps):
    x = dataset[point][0]
    y = dataset[point][1]
    V = 2*(eps**2)
    n = len(dataset)
    within_eps = []
    for i in dataset:
        dist_x = abs(dataset[i][0]-x)
        dist_y = abs(dataset[i][1]-y)
        if dist_x + dist_y <= eps:
            within_eps.append(i)
    k = len(within_eps)
    DE = k/(n*V)
    return DE

##### using fixed k value #####
import numpy as np
def DE_k(dataset, point, k):
    x = dataset[point][0]
    y = dataset[point][1]
    n = len(dataset)
    dist_man = []
    for i in dataset:
        dist_x = abs(dataset[i][0]-x)
        dist_y = abs(dataset[i][1]-y)
        dist_man.append(dist_x + dist_y)
    dist_man = np.sort(dist_man)
    eps = dist_man[k]
    if dist_man[k] == dist_man[k+1]:
        k += 2
    else:
        k += 1
    V = 2*(eps**2)
    DE = k/(n*V)
    return DE

#### crwating table to compare from exercise ####
DE_table_k2 = []
DE_table_k4 = []
DE_table_eps1 = []
DE_table_eps2 = []
for point in dataset:
    DE_table_k2.append(DE_k(dataset, point, 2))
    DE_table_k4.append(DE_k(dataset, point, 4))
    DE_table_eps1.append(DE_eps(dataset, point, 1))
    DE_table_eps2.append(DE_eps(dataset, point, 2))
