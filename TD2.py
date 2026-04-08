'''
Quality measures for cluster
'''

dataset = {1:(10,1), 2:(2,3), 3:(3,4), 4:(1,5), 
           5:(7,7), 6:(6,8), 7:(7,8), 8:(7,9)} 

##### TD2 calculator, for 2 clusters in 2D #####
def TD2_calc(cluster_1, cluster_2, u1, u2):
    u1_x = u1[0]
    u1_y = u1[1]
    TD2_1 = 0
    for point in cluster_1:
        x = point[0]
        y = point[1]
        td2 = abs(u1_x - x)**2+abs(u1_y - y)**2
        TD2_1 += td2
    u2_x = u2[0]
    u2_y = u2[1]
    TD2_2 = 0
    for point in cluster_2:
        x = point[0]
        y = point[1]
        td2 = abs(u2_x - x)**2+abs(u2_y - y)**2
        TD2_2 += td2
    TD2 = TD2_1 + TD2_2
    return TD2_1, TD2_2, TD2
