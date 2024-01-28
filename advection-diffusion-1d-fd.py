# solve u_t + alpha u_x - nu u_xx = 0 with Crank Nicolson method with Dirichlet or periodic BCs
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


plot = True
animate = True
bcs = "periodic"

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
dt = 0.05
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

d = nu*dt/dx**2
c = alpha*dt/dx

P = 10

# time integration
for n in range(1,nt):

    # construct matrices
    A = (c/4 - d/2)*np.eye(nx,k=1) + (d+1)*np.eye(nx) + (-c/4 - d/2)*np.eye(nx,k=-1)
    B = (d/2 - c/4)*np.eye(nx,k=1) + (1-d)*np.eye(nx) + (c/4 + d/2)*np.eye(nx,k=-1)

    if bcs == "periodic":
        # periodic BCs
        A[0,nt-1] = (-c/4 - d/2) # upper right of A
        A[nt-1,0] = (c/4 - d/2) # lower left of A
        B[0,nt-1] = (c/4 + d/2) # upper right of B
        B[nt-1,0] = (d/2 - c/4) # lower left of B

        # analytical solution
        u_analytical[:,n] = -np.sin(np.pi*(x-alpha*t[n]))*(np.exp(-nu*np.pi**2*t[n]))

    Ainv = np.linalg.inv(A)
    C = np.matmul(Ainv,B)

    # numerical approximation
    u_approx[:,n] = np.matmul(C,u_approx[:,n-1])

    if bcs == "dirichlet":
        # dirichlet BCs
        u_approx[0,n] = 0
        u_approx[-1,n] = 0

        # analytical solution
        for i in range(0,nx):
            sinh_sum = 0
            cosh_sum = 0
            for p in range(0,P):
                sinh_sum += ((-1)**p * 2*p * np.sin(p*np.pi*x[i]) * np.exp(-nu*p**2 * np.pi**2 * t[n])) / (alpha**4 + 8 * (alpha*np.pi*nu)**2 * (p**2 + 1) + 16 * (np.pi*nu)**4 * (p**2 - 1)**2)
                cosh_sum += ((-1)**p * (2*p + 1) * np.cos( (2*p+1) * np.pi * x[i]/2 ) * np.exp(-nu * (2*p+1)**2 * np.pi**2 * t[n] / 4) ) / (alpha**4 + (alpha*np.pi*nu)**2 * (8*p**2 + 8*p + 10) + (np.pi*nu)**4 * (4*p**2 + 4*p - 3)**2)
            u_analytical[i,n] = 16*np.pi**2 * nu**3 * alpha * np.exp(alpha * (x[i] - alpha * t[n]/2)/(2*nu)) * (np.sinh(alpha/(2*nu)) * sinh_sum + np.cosh(alpha/(2*nu)) * cosh_sum)

    if plot:
        if n <= 9:
            plt.plot(x,u_approx[:,n],'k.',alpha=n/10)
            plt.plot(x,u_analytical[:,n],'k-',alpha=n/10)
        if n == 10:
            plt.plot(x,u_approx[:,n],'k.',alpha=n/10, label="numerical")
            plt.plot(x,u_analytical[:,n],'k-',alpha=n/10, label="analytical")

if plot:
    plt.xlabel('x')
    plt.ylabel('u')
    plt.grid(True)
    plt.legend(loc="upper right")
    if bcs == "dirichlet":
        plt.title('Advection-diffusion - Dirichlet BCs u(t,1) = 0, u(t,-1) = 0')
        plt.savefig("advdiff-dirichlet.png",dpi=300,format='png',transparent=True)
    if bcs == "periodic":
        plt.title('Advection-diffusion - Periodic BCs u(t,-1) = u(t,1)')
        plt.savefig("advdiff-periodic.png",dpi=300,format='png',transparent=True)
    plt.show()

if animate:
    fig, ax = plt.subplots()

    import matplotlib.animation as animation

    def update(frame):
        ax.cla()
        ax.plot(x,u_approx[:,frame],'k.',label="numerical")
        ax.plot(x,u_analytical[:,frame],'k-',label="analytical")
        ax.set_xlim(-1,1)
        ax.set_ylim(-1,1)
        ax.set_xlabel('x')
        ax.set_ylabel('u')
        ax.legend(loc="upper right")
        if bcs == "dirichlet":
            ax.set_title('Advection-diffusion - Dirichlet BCs u(t,-1) = 0, u(t,1) = 0')
        if bcs == "periodic":
            ax.set_title('Advection-diffusion - Periodic BCs u(t,-1) = u(t,1)')
        return []

    ani = animation.FuncAnimation(fig=fig,func=update,frames=nt,repeat=False)

    fname = "advdiff-" + bcs + ".gif"
    ani.save(filename=fname, writer="pillow")

    plt.show()