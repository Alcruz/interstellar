import numpy as np

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('qtagg')

# Parameters in finance space
r = 0.06  # Risk-free interest rate.
sigma = 0.3  # Volatility.
K = 10  # Strike price.
T = 1  # Maturity date.
delta = 0  # dividends

# Set dicretization method
theta = 0.5

# Compute grid
t_max = 0.5 * np.power(sigma,2) * T  # Set transformed max time
x_min = -2  # Set transformed min price
x_max = 2  # Set transformed max price

dx, dt = 1/250, 1e-3

x_axis = np.arange(x_min, x_max + dx, step=dx)
t_axis = np.arange(0, T+dt, step=dt)
tao_axis = 0.5*np.power(sigma,2)*t_axis
dtao = np.diff(tao_axis)[0]

# Parameters artificail space
q = (2*r) / np.power(sigma, 2)
q_delta = 2*(r-delta) / np.power(sigma, 2)

def payoff(x, tao): return np.exp((tao/4)*(np.power(q_delta-1, 2) + 4*q)) * \
    np.maximum(np.exp((x/2)*(q_delta - 1)) - np.exp((x/2)*(q_delta + 1)), 0)

class GUtil: 
    def __init__(self, x, tao):
        self.x = x
        self.tao = tao
    def __getitem__(self, indx):
         xi, ti = indx
         return payoff(self.x[xi], self.tao[ti])

g = GUtil(x_axis, tao_axis)

# calculate lambda and alpha
lambd = dtao/np.power(dx, 2)
alpha = lambd*theta

# compute initial condition
w = np.empty((tao_axis.size, x_axis.size))
w[0, :] = g[:, 0]

for i in range(tao_axis.size - 1):
    b = np.empty_like(x_axis)
    b[2:-2] = w[i, 2:-2] + lambd*(1-theta)*(w[i, 3:-1] - 2*w[i, 2:-2] + w[i, 1:-3])
    b[1] = w[i, 1] + lambd*(1-theta)*(w[i, 2] -2*w[i, 1] + g[0, i]) + alpha*g[0, i+1]
    b[-2] = w[i, -2] + lambd*(1-theta)*(g[-1, i]-2*w[i, -2] + w[i, -3]) + alpha*g[-1, i+1]

    v = np.maximum(w[i], g[:, i+1])
    eps = 1e-5
    wR = 1
    while 100:
        v_new = np.zeros_like(v)
        for j in range(1, v.size - 1):
            rho = b[j] + alpha*(v_new[j-1] + v[j+1])
            rho /= 1+2*alpha
            v_new[j] = np.maximum(g[j, i+1], v[j] + wR*(rho - v[j]))
        if np.linalg.norm(v_new - v, ord=2) <= eps:
            break
        v = v_new[:]
    w[i+1] = v[:]
    
S = K*np.exp(x_axis)
V = np.empty((11, x_axis.size))
for i in range(11): 
    V[-i-1, :] = K * w[i,  :] * np.exp(-(x_axis/2)*(q_delta - 1)) * np.exp(-tao_axis[i]*((1/4)*np.power(q_delta - 1, 2) + q))

# eps = 1e-5
# i = np.argmax(S[np.abs(V[0, :] + S - K) <= eps]) 
# print(S[i])

plt.plot(np.repeat(S[1:-1, None], 11, axis=1), V[:, 1:-1].T, linewidth=0.5, marker="+", color='black', markersize=4)
plt.xlim((9.7, 10.3))
plt.ylim((0, 0.35))
plt.show()

