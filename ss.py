# %%
import numpy as np
from scipy.stats import rv_continuous, norm
from scipy.special import logsumexp


class SpikeAndSlab(rv_continuous):
    def __init__(self, pi_spike=0.5, spike_std=1e-2, slab_std=1.0, **kwargs):
        super().__init__(**kwargs)
        self.pi_spike = pi_spike
        self.spike_std = spike_std
        self.slab_std = slab_std

        # Underlying distributions
        self.spike_dist = norm(loc=0.0, scale=spike_std)
        self.slab_dist = norm(loc=0.0, scale=slab_std)

    def _pdf(self, x):
        """Mixture PDF."""
        return self.pi_spike * self.spike_dist.pdf(x) + (
            1 - self.pi_spike
        ) * self.slab_dist.pdf(x)

    def _logpdf(self, x, *args, **kwargs):
        args, loc, scale = self._parse_args(*args, **kwargs)
        """Log of mixture PDF."""
        log_components = np.vstack(
            [
                np.log(self.pi_spike) + self.spike_dist.logpdf(x, loc=loc, scale=scale),
                np.log(1 - self.pi_spike)
                + self.slab_dist.logpdf(x, loc=loc, scale=scale),
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


# %%
# Create an instance
spike_slab = SpikeAndSlab(name="spike_slab", pi_spike=0.4, spike_std=1e-1, slab_std=1.0)

# Evaluate logpdf
"""theta = np.array([0.0, 0.5, 2.0])
print(spike_slab.logpdf(theta))

# Draw samples
samples = spike_slab.rvs(size=1000)

# Plot PDF
import matplotlib.pyplot as plt

x = np.linspace(-5, 5, 500)
plt.plot(x, spike_slab.pdf(x), label="Spike-and-Slab PDF")
plt.title("Spike and Slab Distribution")
plt.xlabel(r"$\theta$")
plt.ylabel("Density")
plt.legend()
plt.grid(True)
plt.show()"""

# %%
