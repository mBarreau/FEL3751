# %%
from pinn import PINN
from neural_ode import NeuralODE

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
u = lambda t: tf.concat([tf.sin(t) + tf.cos(t)], 0)

# External simulator
ss = AffineSystem(f, g, h, n=2, std_noise=std_noise, seed=1234)
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
data = (ss.t[:, 0:max_T:k], y[:, 0:max_T:k])

# %% PINN Optimizer
pinn = PINN([20, 20, 20], ss, N_phys=10, T=T + P, seed=1234)
pinn.set_data(data, u)
losses = []
weights = []

# %% Train
loss, weight = pinn.train(3000)
losses += loss
weights += weight

# %% Plot after training
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

plot(ss.t, x, pinn, T=T)
plot(ss.t, y, pinn.y, T=T, name="y")

plt.figure()
error = np.linalg.norm(x - pinn(ss.t).numpy(), axis=0).reshape((1, -1))
plt.plot(ss.t[0, :], error[0, :])
plt.yscale("log")
plt.xlabel("Time [s]")
plt.ylabel("$L_2$ error")
plt.grid()
plt.show()

# %% Neural ODE optimizer (optimize-then-discretize)
neural_ode = NeuralODE([10, 10], ss, seed=1234)
neural_ode.set_data(data, u)
losses = []

# %% Train
losses += neural_ode.train(1000)

# %% Plot after training
plt.figure()
plt.plot(losses)
plt.yscale("log")
plt.xlabel("Epoch")
plt.ylabel("Loss value")
plt.grid()

neural_ode_test = lambda t: neural_ode(t, u=u)
plot(ss.t, x, neural_ode_test, T=T)
plot(ss.t, y, lambda t: neural_ode.y(t, u=u), T=T, name="y")

plt.figure()
error = np.linalg.norm(x - neural_ode_test(ss.t), axis=0).reshape((1, -1))
plt.plot(ss.t[0, :], error[0, :])
plt.yscale("log")
plt.xlabel("Time [s]")
plt.ylabel("$L_2$ error")
plt.grid()
plt.show()

# %%
