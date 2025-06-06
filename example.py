# %%
from pinn import PINN, CPINN

import numpy as np
import matplotlib.pyplot as plt

from utils import plot, AffineSystem

plt.rcParams["text.usetex"] = True
import tensorflow as tf

# %% System
A = np.array([[-5, 3], [0, -2]], dtype=np.float32)
B = np.array([[1], [2]], dtype=np.float32)
C = np.array([[1, 1]], dtype=np.float32)

f = lambda x: A @ x
g = lambda _: B
h = lambda x: C @ x

std_noise = 0.1
u = lambda t: tf.ones_like(t)
# u = lambda t: tf.concat([tf.sin(t) + tf.cos(t)], 0)

# External simulator
ss = AffineSystem(f, g, h, n=2, p=1, q=1, std_noise=std_noise, seed=1234)
T = 6  # training interval
P = 3  # prediction interval
deltaT = 0.01

# External solution
x0 = np.array([[1], [0]], dtype=np.float32)
x = ss.simulate(x0, T + P, deltaT, u=u)
y = ss.y()

# Measurements
k = 10
max_T = int(np.floor(T / deltaT))
data = (ss.t[:, 0:max_T:k], y[:, 0:max_T:k], u(ss.t[:, 0:max_T:k]))

# %% PINN Optimizer
cpinn = CPINN(
    [20, 20, 20], [10, 10, 10], ss, N_phys=10, T=T, P=P, closed_loop=False, seed=1234
)
cpinn.set_data(data)
cpinn.objective(r=u, Q=0.1, R=0.0)
losses = []
weights = []

# %% Train
loss, weight = cpinn.train(5000)
losses += loss
weights += weight

# %% Plot after training
x = ss.simulate(x0, T + P, deltaT, u=cpinn.u)
y = ss.y()

plt.figure()
plt.plot(losses)
plt.yscale("log")
plt.xlabel("Epoch")
plt.ylabel("Loss value")
plt.grid()

plt.figure()
plt.plot(weights)
plt.xlabel("Epoch")
plt.ylabel("Weight value")
plt.grid()

plot(ss.t, x, cpinn, T=T)
plot(ss.t, y, cpinn.y, T=T, name="y")

plt.figure()
plt.plot(ss.t[0, :], cpinn.u(ss.t).numpy().flatten())
plt.xlabel("Time [s]")
plt.ylabel("Control input")
plt.grid()
plt.show()

plt.figure()
error = np.linalg.norm(x - cpinn(ss.t).numpy(), axis=0).reshape((1, -1))
plt.plot(ss.t[0, :], error[0, :])
plt.yscale("log")
plt.xlabel("Time [s]")
plt.ylabel("$L_2$ error")
plt.grid()
plt.show()


# %%
