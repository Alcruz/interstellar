import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('qtagg')

r = 0.1  # Risk-free interest rate.
sigma = 0.2  # Volatility.
K = 1  # Strike price.
T = 1  # Maturity date.
x_inf = 2  # Front-fixing infinity.

M = 999  # Number of internal nodes in spatial direction.
N = 1999999  # Number of internal nodes in time direction.

dt = T / (N + 1)
dx = (x_inf - 1) / (M + 1)
p = np.zeros(M + 1)

# lambd_ = ((M+1)**2/(N+1)) * (T / (x_inf - 1))
# phi = (((M+1)/(N+1)) * (T / (x_inf - 1)))
x = np.linspace(1, x_inf, num=M+1)

A = 0.5 * np.power(sigma, 2) * np.power(x, 2) * (dt / np.power(dx, 2))
A -= x * (r - (1/dt)) * (dt/(2*dx))

B = 1 - np.power(sigma, 2) * np.power(x, 2) * (dt / np.power(dx, 2))
B -= r*dt

C = 0.5 * np.power(sigma, 2) * np.power(x, 2) * (dt / np.power(dx, 2))
C += x * (r - (1/dt)) * (dt/(2*dx))

S_bar = K
for n in range(N, -1, -1):
    D = x/(2*dx)
    D[1:-1] *= (p[2:] - p[:-2])/S_bar

    S_bar = K - (A[1]*p[0] + B[1]*p[1] + C[1]*p[2])
    S_bar /= D[1] + 1 + dx

    p[0] = K - S_bar
    p[1] = K - (1+dx)*S_bar
    p[2:-1] = A[2:-1]*p[1:-2] + B[2:-1]*p[2:-1] + C[2:-1]*p[3:] + D[2:-1]*S_bar

print(S_bar)
print(np.max(p))
plt.plot(x, p)
plt.xlim(1, 2)
plt.ylim(bottom=0)
plt.show()
