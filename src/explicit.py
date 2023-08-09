import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import datetime

matplotlib.use('qtagg')

r = 0.1  # Risk-free interest rate.
sigma = 0.2  # Volatility.
K = 1.0  # Strike price.
T = 1.0  # Maturity date.
x_inf = 2.0  # Front-fixing infinity.

M = 99 # Number of internal nodes in spatial direction.
N = 1999  # Number of internal nodes in time direction.

dt, dt_inv = T/(N+1), (N+1)/T
dx, dx_inv = (x_inf-1)/(M+1), (M+1)/(x_inf - 1)

gamma = (np.power(M+1, 2)/(N+1)) * (T/np.power(x_inf - 1, 2))
alpha = 0.5 * ((M+1)/(N+1)) * (T/(x_inf - 1))

x = np.arange(1, 2+dx, dx)

A = 0.5 * np.power(sigma*x, 2) * gamma - x * (r - dt_inv) * alpha
B = 1 - np.power(sigma*x, 2) * gamma - r*dt
C = 0.5 * np.power(sigma*x, 2) * gamma + x * (r - dt_inv) * alpha

starting_time = datetime.datetime.now()
print("Start time:", starting_time)
p = np.zeros(M+2)
S_bar = K
for n in range(N, -1, -1):
    D = 0.5*x*dx_inv
    D[1:-1] *= (p[2:]-p[:-2]) * (1/S_bar)

    S_bar = K - (A[1]*p[0] + B[1]*p[1] + C[1]*p[2])
    S_bar /= D[1] + 1 + dx
    if (n % 100) == 0:
        cur_time = datetime.datetime.now()
        print("Elapsed seconds since last iteration:", (cur_time - starting_time).seconds)
        print(f"Iteration {n}:", S_bar)
    p[2:-1] = A[2:-1]*p[1:-2] + B[2:-1]*p[2:-1] + C[2:-1]*p[3:] + D[2:-1]*S_bar
    p[0] = K - S_bar
    p[1] = K - (1+dx)*S_bar

print(S_bar)
S = S_bar * np.arange(0, x_inf+dx, step=dx) 
P = np.concatenate([K - S[S < S_bar], p])

plt.plot(S, np.maximum(K-S, 0))
plt.plot(S, P)
plt.vlines(S_bar, 0, p[0], linestyles='dashed')
plt.xlim(0, S_bar*x_inf)
plt.ylim(bottom=0, top=K)
plt.show()
