# %%
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["text.usetex"] = True
from scipy.integrate import solve_ivp
from scipy.stats import norm, laplace
from ss import spike_slab
from scipy.stats.kde import gaussian_kde
import emcee


# Dynamical function f for the ODE with parameters theta
def f(t, x, theta):
    return theta[0] * x + theta[1] * x**2 + theta[2] * x**3


# Generate synthetic data
theta_true = np.array([-2.0, 0, -0.1])
sigma = 0.1
x0 = 1.0
t_obs = np.linspace(0, 2, 40)
y_true = solve_ivp(lambda t, x: f(t, x, theta_true), [0, 2], [x0], t_eval=t_obs).y[0]
y_obs = y_true + np.random.normal(0, sigma, len(t_obs))

plt.figure()
plt.plot(t_obs, y_obs, "o", label="Observed data")
plt.plot(t_obs, y_true, "r-", label="True trajectory")
plt.xlabel("Time")
plt.ylabel("$y$")
plt.grid()
plt.legend()
plt.show()


# %%
# Log-likelihood function
def log_likelihood(theta, t_obs, y_obs, sigma):
    loglike = 0.0
    y_pred = solve_ivp(lambda t, x: f(t, x, theta), [0, 2], [x0], t_eval=t_obs).y[0]
    for s in zip(y_pred, y_obs):
        loglike += norm.logpdf(s[1], loc=s[0], scale=sigma) + sum(
            [laplace.logpdf(theta[i], scale=5) for i in range(len(theta))]
        )
    return loglike


# MCMC setup
def log_prob(theta_array):
    return log_likelihood(theta_array, t_obs, y_obs, sigma)


ndim = len(theta_true)
nwalkers = 10
initial_pos = [theta_true + 1e-4 * np.random.randn(ndim) for _ in range(nwalkers)]

sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)
sampler.run_mcmc(initial_pos, 3000, progress=True)
samples = sampler.get_chain(discard=500, flat=True)  # Discard burn-in and flatten

# %% Plot the results
fig, axs = plt.subplots(ndim, sharex=True, figsize=(7, 3))
for i in range(ndim):
    kde = gaussian_kde(samples[:, i])
    x = np.linspace(min(samples[:, i]), max(samples[:, i]), 100)
    axs[i].plot(x, kde(x))
    axs[i].grid()
    # axs[i].set_xlabel(rf"$\theta_{i}$")
    y_lim = axs[i].get_ylim()
    axs[i].vlines(
        theta_true[i], y_lim[0], y_lim[1], color="r", linestyle="--", label="True value"
    )
plt.legend

# %%
