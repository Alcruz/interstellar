import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse import diags
from scipy.optimize import root

r = 0.1  # Risk-free interest rate.
sigma = 0.2  # Volatility.
K = 1.0  # Strike price.
T = 1.0  # Maturity date.
x_inf = 2.0  # Front-fixing infinity.

M = 99  # Number of internal nodes in spatial direction.
N = 99  # Number of internal nodes in time direction.

dt, dt_inv = T/(N+1), (N+1)/T
dx, dx_inv = (x_inf-1)/(M+1), (M+1)/(x_inf - 1)

lambd = (np.power(M+1, 2)/(N+1)) * (T/np.power(x_inf - 1, 2))
kappa = 0.5*((M+1)/(N+1)) * (T/(x_inf - 1))

x = np.arange(1, 2+dx, dx)

alpha = 1 + lambd*np.power(sigma*x, 2) + r*dt

def beta(s): 
    return -0.5*lambd*np.power(sigma,2)*np.power(x,2) + kappa*x*(r - (S_bar - s)/(dt*s))

def gamma(s): 
    return -0.5*lambd*np.power(sigma,2)*np.power(x,2) - kappa*x*(r - (S_bar - s)/(dt*s))

def F(y, b, _):
    p, s, = y[:-1], y[-1]
    _beta = beta(s)
    _gamma = gamma(s)
    
    A = diags([_beta[3:-1], alpha[2:-1], _gamma[1:-2]], [-2, -1, 0], shape=(M, M-1)).toarray()
    f = b[1:-1]
    f[0] -= _beta[1]*(K-s) + alpha[1]*((K-(1+dx)*s))
    f[1] -= _beta[2]*(K-(1+dx)*s)
    res = A@p - f
    return res

def J(y, _, S_bar):
    p, s, = y[:-1], y[-1]
    _beta = beta(s)
    _gamma = gamma(s)
    
    dgamma_ds = - 0.5 * dx_inv * x * S_bar / s**2
    dbeta_ds = 0.5 * dx_inv * x * S_bar / s**2 

    retVal = diags([_beta[3:-1], alpha[2:-1], _gamma[1:-1]], [-2, -1, 0], shape=(M,M)).toarray()

    retVal[-1, :] = 0
    retVal[:, -1] = 0

    retVal[0, -1] = dgamma_ds[1]*p[0] + dbeta_ds[1]*(K - S_bar) - _beta[1] - alpha[1]*(1+dx)
    retVal[1, -1] = dgamma_ds[2]*p[1] + dbeta_ds[2]*(K - (1+dx)*S_bar) - _beta[2]*(1+dx)
    retVal[2:-1, -1] = dbeta_ds[3:-2]*p[:-2] + dgamma_ds[3:-2]*p[2:]

    retVal[-1, -3] = _beta[-2]
    retVal[-1, -2] = alpha[-2]
    retVal[-1, -1] = dbeta_ds[-2]*p[-2]

    return retVal

S_bar = K
p = np.zeros((M+2))
for _ in range(N, -1, -1):
    sol = root(lambda y: F(y, np.copy(p[:]), S_bar), np.concatenate([p[2:-1], [S_bar]]), jac=lambda y: J(y, np.copy(p[:]), S_bar), method='hybr')
    *p[2:-1], S_bar = sol['x']
    p[0] = K - S_bar
    p[1] = K - (1+dx)*S_bar

print(S_bar)
S = S_bar * np.arange(0, x_inf+dx, step=dx) 
P = np.concatenate([K - S[S < S_bar], p])

plt.plot(S, np.maximum(K-S, 0))
plt.plot(S, P)
plt.vlines(S_bar, 0, p[0])
plt.xlim(0, S_bar*x_inf)
plt.ylim(bottom=0, top=K)
plt.show()
