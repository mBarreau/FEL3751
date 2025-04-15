import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import numpy as np


def plot(t, x, x_hat, T=None, subsampling=10, name="x"):
    colors = cm.rainbow(np.linspace(0, 1, x.shape[0]))
    fig, ax = plt.subplots()
    x_hat_values = x_hat(t).numpy()
    for i, c in enumerate(colors):
        plt.plot(t[0, :], x[i, :], color=c, label=rf"${name}_{i+1}$", alpha=0.3)
        plt.scatter(
            t[0, 0::subsampling],
            x_hat_values[i, 0::subsampling],
            s=5,
            color=c,
            label=rf"$\hat {name}_{i+1}$",
        )
    plt.grid()
    plt.xlabel("Time [s]")
    plt.ylabel(f"${name}$")
    plt.legend()

    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    if T is not None:
        ax.fill_betweenx(ylim, 0, T, alpha=0.1, color="gray")
        ax.set_ylim(ylim)
    ax.set_xlim([0, xlim[1]])

    plt.show(block=False)
