# solve u_t + alpha u_x - nu u_xx = 0 with 2nd order forward FD in space and RK4 in time
# alpha = 1
# nu = 0.1

import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import matplotlib.pyplot as plt
import numpy as np
import scipy

from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Times']
rcParams['font.size'] = 16
rcParams["figure.figsize"] = (8,4)
rcParams['figure.dpi']= 200
rcParams["figure.autolayout"] = True


# physical parameters
alpha = 1 # advection
nu = 0.1 # diffusion

# spatial mesh
dx = 0.05
x0 = -1
xf = 1
x = np.arange(x0,xf+dx,dx)
nx = len(x)

# temporal mesh
dt = 0.01
t0 = 0
tf = 2
t = np.arange(t0,tf+dt,dt)
nt = len(t)

# initial condition
u0 = -np.sin(np.pi*x)

# solution structures
u_approx = np.empty((nx,nt))
u_analytical = np.empty((nx,nt))
u_approx[:,0] = u0
u_analytical[:,0] = u0

# plot initial condition
plt.plot(x,u_approx[:,0],'k.',alpha=0.1)
plt.plot(x,u_analytical[:,0],'k',alpha=0.1)

def f(u,t):
    A = (nu - 0.5*alpha*dx)*np.eye(nx-2,k=1) + (-2*nu)*np.eye(nx-2) + (nu + 0.5*alpha*dx)*np.eye(nx-2,k=-1)
    return np.matmul(A,u)/dx**2

# time integration
for n in range(0,nt-1):

    # RK4 time-stepping
    k1 = f(u_approx[1:-1,n], t[n])
    k2 = f(u_approx[1:-1,n] + k1*dt/2, t[n] + dt/2)
    k3 = f(u_approx[1:-1,n] + k2*dt/2, t[n] + dt/2)
    k4 = f(u_approx[1:-1,n] + k3*dt, t[n] + dt)
    u_approx[1:-1,n+1] = u_approx[1:-1,n] + (k1 + 2*k2 + 2*k3 + k4)*dt/6

    # dirichlet BCs
    u_approx[0,n+1] = 0
    u_approx[-1,n+1] = 0

    # analytical solution
    for i in range(0,nx):
        sinh_sum = 0
        cosh_sum = 0
        for p in range(0,20):
            sinh_sum += ((-1)**p * 2*p * np.sin(p*np.pi*x[i]) * np.exp(-nu*p**2 * np.pi**2 * t[n])) / (alpha**4 + 8 * (alpha*np.pi*nu)**2 * (p**2 + 1) + 16 * (np.pi*nu)**4 * (p**2 - 1)**2)
            cosh_sum += ((-1)**p * (2*p + 1) * np.cos( (2*p+1) * np.pi * x[i]/2 ) * np.exp(-nu * (2*p+1)**2 * np.pi**2 * t[n] / 4) ) / (alpha**4 + (alpha*np.pi*nu)**2 * (8*p**2 + 8*p + 10) + (np.pi*nu)**4 * (4*p**2 + 4*p - 3)**2)
        u_analytical[i,n+1] = 16*np.pi**2 * nu**3 * alpha * np.exp(alpha * (x[i] - alpha * t[n]/2)/(2*nu)) * (np.sinh(alpha/(2*nu)) * sinh_sum + np.cosh(alpha/(2*nu)) * cosh_sum)

    if n%10 == 0 and n < 50:
        plt.plot(x,u_approx[:,n],'k.',alpha=n/50)
        plt.plot(x,u_analytical[:,n],'k-',alpha=n/50)
    if n == 50:
        plt.plot(x,u_approx[:,n],'k.',alpha=n/50, label="numerical")
        plt.plot(x,u_analytical[:,n],'k-',alpha=n/50, label="analytical")

plt.xlabel('x')
plt.ylabel('u')
plt.grid(True)
plt.legend(loc="upper right")
plt.title('1D space-dependent advection-diffusion')
plt.savefig("advdiff-dirichlet.png",dpi=300,format='png',transparent=True)
plt.show()

