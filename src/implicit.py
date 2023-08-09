import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse import diags
from scipy.optimize import root

import datetime

import numpy as np

def solve(F, J, x0, epsilon=1e-6, max_iterations=100):
    """
    Newton's method for solving a system of nonlinear equations.
    
    Arguments:
        f: A function that takes a vector x and returns a vector of equations evaluated at x.
        J: A function that takes a vector x and returns the Jacobian matrix of f evaluated at x.
        x0: The initial guess for the solution vector.
        epsilon: The desired level of accuracy.
        max_iterations: The maximum number of iterations.
    
    Returns:
        The approximate solution vector.
    """
    x = x0
    for i in range(max_iterations):
        f_val = F(x)
        if np.linalg.norm(f_val) < epsilon:
            return x
        J_val = J(x)
        if np.linalg.det(J_val) == 0:
            break
        delta_x = np.linalg.solve(J_val, -f_val)
        x = x + delta_x
    return x


r = 0.1  # Risk-free interest rate.
sigma = 0.2  # Volatility.
K = 1.0  # Strike price.
T = 1.0  # Maturity date.
x_inf = 2.0  # Front-fixing infinity.
delta = 0 # dividends

# dt, dx = 0.001, 0.001
dt, dx = 5e-2, 0.1

N = int(T/dt)
M = int((x_inf-1)/dx)


lambd = np.round(dt / np.power(dx,2))
kappa = dt / dx

x = np.arange(1, x_inf+dx, dx)

theta = 0

alpha = 1 + theta*(lambd*np.power(sigma*x, 2) + r*dt)

def beta(s): 
    return theta*(-0.5*lambd*np.power(sigma,2)*np.power(x,2) + 0.5*kappa*x*((r-delta) - (S_bar - s)/(dt*s)))

def gamma(s): 
    return theta*(-0.5*lambd*np.power(sigma,2)*np.power(x,2) - 0.5*kappa*x*((r-delta) - (S_bar - s)/(dt*s)))

def dbeta_ds(s): 
    return  theta*(0.5*(x/dx)*S_bar / s**2) 

def dgamma_ds(s): 
    return  theta*(-0.5*(x/dx)*S_bar / s**2)

a = 1 + (1-theta)*(- lambd*np.power(sigma, 2)*np.power(x, 2) - r*dt)
b = (1-theta)*(0.5*lambd*np.power(sigma, 2)*np.power(x, 2) - 0.5*x*kappa*((r-delta) - (1/dt)))
c = (1-theta)*(0.5*lambd*np.power(sigma, 2)*np.power(x, 2) + 0.5*x*kappa*((r-delta) - (1/dt)))
d = (1-theta)*(0.5*x/dx)

def F(y, v, S_bar):
    p, s = y[:-1], y[-1]
    _beta = beta(s)
    _gamma = gamma(s)
    
    A = diags([_beta[3:-1], alpha[2:-1], _gamma[1:-2]], [-2, -1, 0], shape=(M-1, M-2))
    f = b[1:-1]*v[:-2] + a[1:-1]*v[1:-1] + c[1:-1]*v[2:] + d[1:-1]*(v[2:] - v[:-2])*s/S_bar
    f[0] -= _beta[1]*(K-s) + alpha[1]*((K-(1+dx)*s))
    f[1] -= _beta[2]*(K-(1+dx)*s)
    res = A@p - f
    return res

def J(y, v, S_bar):
    p, s, = y[:-1], y[-1]
    _beta = beta(s)
    _gamma = gamma(s)
    
    _dgamma_ds = dgamma_ds(s)
    _dbeta_ds = dbeta_ds(s)

    retVal = diags([_beta[3:-1], alpha[2:-1], _gamma[1:-1]], [-2, -1, 0], shape=(M-1,M-1)).toarray()

    retVal[-1, :] = 0
    retVal[:, -1] = -d[1:-1]*(v[2:] - v[:-2])/S_bar

    retVal[0, -1] = _dgamma_ds[1]*p[0] + _dbeta_ds[1]*(K - S_bar) - _beta[1] - alpha[1]*(1+dx)
    retVal[1, -1] = _dgamma_ds[2]*p[1] + _dbeta_ds[2]*(K - (1+dx)*S_bar) - _beta[2]*(1+dx)
    retVal[2:-1, -1] = _dbeta_ds[3:-2]*p[:-2] + _dgamma_ds[3:-2]*p[2:]

    retVal[-1, -3] = _beta[-2]
    retVal[-1, -2] = alpha[-2]
    retVal[-1, -1] = _dbeta_ds[-2]*p[-2]
    return retVal


starting_time = datetime.datetime.now()
print("Start time:", starting_time)
S_bar = s = K
v = np.zeros_like(x)
for n in range(N-1, -1, -1):
    # *v[2:-1], S_bar = root(
    #     fun=lambda y: F(y, np.copy(v[:]), S_bar), 
    #     # jac=lambda y: J(y, np.copy(v[:]), S_bar), 
    #     x0=np.concatenate([v[2:-1], [S_bar]]),
    #     tol=1e-18
    #     # max_iterations=10
    # ).x
    *v[2:-1], S_bar = solve(
        F=lambda y: F(y, np.copy(v[:]), S_bar), 
        J=lambda y: J(y, np.copy(v[:]), S_bar), 
        x0=np.concatenate([v[2:-1], [S_bar]]),
        epsilon=1e-18,
        max_iterations=100
    )
    if (n % 100) == 0:
        cur_time = datetime.datetime.now()
        print("Elapsed seconds since last iteration:", (cur_time - starting_time).seconds)
        print(f"Iteration {n}:", S_bar)
    v[0] = K - S_bar
    v[1] = K - (1+dx)*S_bar

end = datetime.datetime.now()
print("Final time:", end)
print("S_bar:", S_bar)
S = S_bar * np.arange(0, x_inf+dx, step=dx) 
P = np.concatenate([K - S[S < S_bar], v])

plt.plot(S, np.maximum(K-S, 0))
plt.plot(S, P)
plt.vlines(S_bar, 0, v[0], linestyles='dashed')
plt.xlim(0, S_bar*x_inf)
plt.ylim(bottom=0, top=K)
plt.show()
