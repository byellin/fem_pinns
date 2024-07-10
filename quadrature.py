import numpy as np
#import scipy as sp

P0 = np.array([0,0])

aux1, aux2 = 1/np.sqrt(3), np.sqrt(3/5)
aux3 = np.sqrt(14/15)

P1= np.array([aux1,aux2])
P2= np.array([-aux1,aux2])
P3= np.array([aux1,-aux2])
P4= np.array([-aux1,-aux2])

P5 = np.array([aux3,0])
P6 = np.array([-aux3,0])

weights = np.array([8/7,20/36,20/36,20/36,20/36,20/63,20/63])

def f(P):
    x,y=P[0],P[1]
    return x**4+y**4

resA = weights[0]*f(P0)
resB = weights[1]*f(P1)+ weights[2]*f(P2)+ weights[3]*f(P3)+ weights[4]*f(P4)
resC = weights[5]*f(P5) + weights[6]*f(P6)

res = resA + resB + resC

print(res)