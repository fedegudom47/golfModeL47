'''
Copied Chandler's R notes into Python
'''
import numpy as np
import matplotlib.pyplot as plt

# 📌 RBF Kernel Function (Radial Basis Function or Gaussian Kernel)
# This function returns a scalar value representing the similarity between two input vectors x and y.
# The similarity decreases with the distance between x and y, controlled by the hyperparameter sigma (the length scale).
# This is a core building block of the GP's prior covariance structure.
def rbf_kernel(x, y, sigma):
    """
    RBF Kernel: k(x, y) = exp(-||x - y||^2 / (2 * sigma^2))

    Parameters:
    - x, y: input vectors (usually scalars here)
    - sigma: kernel bandwidth / length scale (controls smoothness)

    Returns:
    - kernel value (scalar)
    """
    return np.exp(-np.sum((x - y) ** 2) / (2 * sigma ** 2))


# 🧠 "True" function - the underlying function we want to learn
# This represents the ground truth used to generate data, which in real scenarios is unknown.
def computer_simulator(x):
    """
    Synthetic function to simulate noisy observations from.

    y = log(x + 0.1) + sin(5πx)

    Nonlinear with periodicity and smoothness — a good test case for GPR.
    """
    return np.log(x + 0.1) + np.sin(5 * np.pi * x)


# ⚙️ Core Gaussian Process Regression function
# This function takes in training data, a test grid, and computes the posterior predictive distribution over test points.
def gpreg(x, y, lam, sig, design):
    """
    Perform Gaussian Process Regression with RBF kernel.

    Parameters:
    - x: observed inputs (shape: n)
    - y: observed noisy outputs (shape: n)
    - lam: RBF length scale parameter
    - sig: standard deviation of noise in y
    - design: new input locations to predict at (shape: m)

    Returns:
    - mean: predicted posterior mean at design points (shape: m)
    - vars: predicted posterior variances (shape: m)
    """

    # Number of observed training points and test (design) points
    n = len(x)
    m = len(design)

    # Combine both training and prediction points into one array so we can build a full covariance matrix
    all_points = np.concatenate((x, design))  # shape: (n + m,)
    N = n + m

    # Step 1: Compute the full N x N prior covariance matrix
    # This represents the joint prior over both training and prediction points
    Sigma = np.zeros((N, N))
    for i in range(N):
        for j in range(i, N):
            # Compute kernel between point i and j
            # np.array([..]) ensures we treat each value as a 1D vector, which makes kernel computations generalisable
            k_val = rbf_kernel(np.array([all_points[i]]), np.array([all_points[j]]), lam)
            Sigma[i, j] = k_val
            Sigma[j, i] = k_val  # Symmetry: kernel matrices are symmetric

    # Step 2: Partition Sigma into 4 blocks:
    #
    #    [ S11 | S12 ]
    #    [ S21 | S22 ]
    #
    # S11 = Cov(train, train)
    # S12 = Cov(train, test)
    # S21 = Cov(test, train)
    # S22 = Cov(test, test)
    S11 = Sigma[:n, :n]
    S12 = Sigma[:n, n:]
    S21 = Sigma[n:, :n]
    S22 = Sigma[n:, n:]

    # Step 3: Compute the posterior predictive distribution using closed-form GPR formulas:
    #
    # μ* = S21 · (S11 + σ²I)⁻¹ · y
    # Σ* = S22 - S21 · (S11 + σ²I)⁻¹ · S12
    #
    # This gives the GP posterior over the test points (design)

    # Add noise variance to training covariance matrix (Gaussian likelihood)
    # This regularises the inversion and models uncertainty in y
    noise_matrix = sig**2 * np.eye(n)

    # Compute the posterior mean vector
    # This gives the predicted output at the design points
    inv = S21 @ np.linalg.inv(S11 + noise_matrix)
    mean = inv @ y  # Shape: (m,)

    # Compute the posterior covariance matrix for the predictions
    # From this we extract just the diagonal (variances)
    cov = S22 - inv @ S12
    vars = np.diag(cov)  # Extract the uncertainty (variance) for each design point

    return mean, vars


# 🔧 DATA GENERATION & MODEL FITTING

# Reproducibility
np.random.seed(42)

# 1. Generate training data (x, y)
n = 10
x = np.sort(np.random.rand(n))  # 10 random points in [0, 1]
sig = 0.1  # Noise standard deviation
y = computer_simulator(x) + np.random.normal(0, sig, n)  # Add noise to observations

# 2. Define test points (the "design" points where we want predictions)
design = np.linspace(0, 1, 101)  # Grid of 101 values between 0 and 1
truth = computer_simulator(design)  # True function values for reference

# 3. Fit GPR model
mean, vars = gpreg(x, y, lam=0.1, sig=sig, design=design)

# 📊 PLOTTING THE RESULTS

plt.figure(figsize=(10, 6))

# Posterior mean prediction
plt.plot(design, mean, label='GP Mean', linewidth=2)

# Uncertainty band: ±2 standard deviations (approx. 95% confidence interval)
plt.fill_between(design,
                 mean - 2 * np.sqrt(vars),
                 mean + 2 * np.sqrt(vars),
                 color='blue', alpha=0.2,
                 label='±2 SD')

# Ground truth (the "oracle" function)
plt.plot(design, truth, 'r--', label='Truth', linewidth=1.5)

# Training data points (noisy observations)
plt.scatter(x, y, color='black', label='Observations', zorder=5)

plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Gaussian Process Regression (No Libraries)')
plt.grid(True)
plt.tight_layout()
plt.show()
