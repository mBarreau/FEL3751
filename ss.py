# %%
import numpy as np
from scipy.stats import rv_continuous, norm
from scipy.special import logsumexp


class SpikeAndSlab(rv_continuous):
    def __init__(self, pi_spike=0.5, spike_std=1e-2, **kwargs):
        super().__init__(**kwargs)
        self.name = "spike_slab"
        self.pi_spike = pi_spike
        self.spike_std = spike_std

        # Underlying distributions
        self.spike_dist = norm(loc=0.0, scale=spike_std)
        self.slab_dist = norm

    def _pdf(self, x, *args, **kwargs):
        """Mixture PDF."""
        args, loc, scale = self._parse_args(*args, **kwargs)
        return self.pi_spike * self.spike_dist.pdf(x) + (
            1 - self.pi_spike
        ) * self.slab_dist.pdf(x, scale=scale)

    def _logpdf(self, x, *args, **kwargs):
        args, loc, scale = self._parse_args(*args, **kwargs)
        """Log of mixture PDF."""
        log_components = np.vstack(
            [
                np.log(self.pi_spike) + self.spike_dist.logpdf(x),
                np.log(1 - self.pi_spike) + self.slab_dist.logpdf(x, scale=scale),
            ]
        )
        return logsumexp(log_components, axis=0)

    def _rvs(self, size=None, random_state=None):
        """Random sampling."""
        rng = self._random_state if random_state is None else random_state
        mixture = rng.uniform(0, 1, size=size) < self.pi_spike
        samples = np.zeros(size)
        samples[mixture] = self.spike_dist.rvs(size=np.sum(mixture), random_state=rng)
        samples[~mixture] = self.slab_dist.rvs(size=np.sum(~mixture), random_state=rng)
        return samples


spike_slab = SpikeAndSlab(pi_spike=0.4, spike_std=1e-1)

# %%
"""from scipy.stats import norm, laplace

sigma = 10
# Create an instance
dist_prior = [laplace, spike_slab]

# Evaluate logpdf
theta = np.array([0.0, 0.5, 2.0])
print(spike_slab.logpdf(theta))

# Draw sample

# Plot PDF
import matplotlib.pyplot as plt

x = np.linspace(-3 * sigma, 3 * sigma, 500)
for prior in dist_prior:
    samples = prior.rvs(size=2000, scale=sigma)
    plt.hist(samples, density=True, bins=30, label=prior.name, alpha=0.5)
    plt.plot(x, prior.pdf(x, scale=sigma), label=f"{prior.name} PDF")
    plt.xlabel(r"$\theta$")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    plt.show()"""

# %%
