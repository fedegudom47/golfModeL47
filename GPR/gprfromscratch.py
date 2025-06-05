import numpy as np
import matplotlib.pyplot as plt

# RBF Kernel
def rbf_kernel(x, y, sigma):
    return np.exp(-np.sum((x - y) ** 2) / (2 * sigma ** 2))

# Simulator function (the "true" function)
def computer_simulator(x):
    return np.log(x + 0.1) + np.sin(5 * np.pi * x)

# Gaussian Process Regression function
def gpreg(x, y, lam, sig, design):
    n = len(x)
    m = len(design)
    all_points = np.concatenate((x, design))
    N = n + m
    
    # Compute full kernel matrix
    Sigma = np.zeros((N, N))
    for i in range(N):
        for j in range(i, N):
            k_val = rbf_kernel(np.array([all_points[i]]), np.array([all_points[j]]), lam)
            Sigma[i, j] = k_val
            Sigma[j, i] = k_val  # symmetry

    S11 = Sigma[:n, :n]
    S12 = Sigma[:n, n:]
    S21 = Sigma[n:, :n]
    S22 = Sigma[n:, n:]
    
    inv = S21 @ np.linalg.inv(S11 + sig**2 * np.eye(n))
    mean = inv @ y
    cov = S22 - inv @ S12
    vars = np.diag(cov)
    
    return mean, vars

# Sample usage
np.random.seed(42)
n = 10
x = np.sort(np.random.rand(n))
sig = 0.1
y = computer_simulator(x) + np.random.normal(0, sig, n)

design = np.linspace(0, 1, 101)
truth = computer_simulator(design)

mean, vars = gpreg(x, y, lam=0.1, sig=sig, design=design)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(design, mean, label='GP Mean')
plt.fill_between(design, mean - 2 * np.sqrt(vars), mean + 2 * np.sqrt(vars), color='blue', alpha=0.2, label='±2 SD')
plt.plot(design, truth, 'r--', label='Truth')
plt.scatter(x, y, color='black', label='Observations')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Gaussian Process Regression (No Libraries)')
plt.grid(True)
plt.show()
