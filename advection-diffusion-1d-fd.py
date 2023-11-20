# solve u_t + alpha u_x - nu u_xx = 0 with Crank Nicolson method
# alpha = 1
# nu = 0.1

import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import matplotlib.pyplot as plt
import numpy as np
import math
import scipy

from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Times']
rcParams['font.size'] = 16
rcParams["figure.figsize"] = (8,4)
rcParams['figure.dpi']= 200
rcParams["figure.autolayout"] = True

dx = 0.01
dt = 0.01
t0 = 0
x0 = 0
tf = 1
xf = 1

X = np.arange(x0,xf+dx,dx)
T = np.arange(t0,tf+dt,dt)

N = len(T)
I = len(X)

u0 = np.exp(-100*(X[1:-1] - 0.5)**2)
u = np.empty((N+2,I))
u[0,0] = 0
u[0,1:-1] = u0
u[0,-1] = 0
plt.plot(X,u[0,:],'k',alpha=0.1)
alpha = 1
nu = 0.1
d = nu*dt/dx**2
c = alpha*dt/dx

A = (c/4 - d/2)*np.eye(I,k=1) + (d+1)*np.eye(I) + (-c/4 - d/2)*np.eye(I,k=-1)
B = (d/2 - c/4)*np.eye(I,k=1) + (1-d)*np.eye(I) + (c/4 + d/2)*np.eye(I,k=-1)

for n in range(1,N):
    u[n,:] = np.matmul(np.matmul(np.linalg.inv(A),B),u[n-1,:])
    u[n,0] = 0
    u[n,-1] = 0
    if n <= 5:
        plt.plot(X,u[n,:],'k',alpha=n/5)

plt.title('Space-dependent advection-diffusion')
plt.xlabel('x')
plt.ylabel('u')
plt.show()

