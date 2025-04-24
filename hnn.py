# %%
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

plt.rcParams["text.usetex"] = True
from scipy.integrate import solve_ivp
from utils import NeuralNetwork


# Generate synthetic data


def true_pendulum(t, y):
    q, p = y
    dq = p
    dp = -np.sin(q)
    return [dq, dp]


def generate_pendulum_data(n_samples=1000):
    q = tf.random.uniform((n_samples, 1), minval=-np.pi, maxval=np.pi)
    p = tf.random.uniform((n_samples, 1), minval=-2.0, maxval=2.0)
    q_p = tf.concat([q, p], axis=1)

    dq, dp = true_pendulum(0, (q, p))
    dq_dp = tf.concat([dq, dp], axis=1)

    return q_p, dq_dp


q_p_train, dq_dp_train = generate_pendulum_data(n_samples=10000)


# %%
# Define the HNN


class Flow(tf.Module):
    def __init__(self, layers, seed=1234):
        super().__init__()
        self.nn = NeuralNetwork(layers, seed=seed)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    def __call__(self, x):
        return tf.transpose(self.nn(x))

    def loss_fn(self, q_p, dq_dp_true):
        dq_dp_pred = self(q_p)
        return tf.reduce_mean(tf.square(dq_dp_pred - dq_dp_true))

    def _ode_fn(self, t, y):
        y_tf = tf.convert_to_tensor(y.reshape(1, self.nn.layers[0]), dtype=tf.float32)
        dy_tf = self(y_tf)
        return dy_tf.numpy().flatten()

    def trajectory(self, t_span, y0, t_eval):
        sol = solve_ivp(self._ode_fn, t_span, y0, t_eval=t_eval)
        return sol.y.T

    @tf.function
    def update(self, q_p, dq_dp):
        with tf.GradientTape() as tape:
            loss = self.loss_fn(q_p, dq_dp)
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return loss


class HamiltonianNN(Flow):
    def __init__(self, layers, seed):
        super().__init__(layers, seed)

    def value(self, inputs):
        return self.nn(inputs)

    def __call__(self, q_p):
        with tf.GradientTape() as tape:
            tape.watch(q_p)
            H = self.nn(q_p)
        grad = tape.gradient(H, q_p)
        dH_dq = grad[:, 0:1]
        dH_dp = grad[:, 1:2]
        dq_dt = dH_dp
        dp_dt = -dH_dq
        return tf.concat([dq_dt, dp_dt], axis=1)


# Initialize model and optimizer
hnn = HamiltonianNN([2, 20, 20, 1], seed=1234)
nn = Flow([2, 20, 20, 2], seed=1234)


# %%
# Training loop
pbar = tqdm(range(5000))
for i in pbar:
    loss = hnn.update(q_p_train, dq_dp_train).numpy()
    pbar.set_description(f"Loss: {loss:.6f}")

pbar = tqdm(range(5000))
for i in pbar:
    loss = nn.update(q_p_train, dq_dp_train).numpy()
    pbar.set_description(f"Loss: {loss:.6f}")


# %% Plots
t_eval = np.linspace(0, 200, 500)
y0 = [1.0, 0.0]  # initial state: q=1 rad, p=0

true_traj = solve_ivp(true_pendulum, [t_eval[0], t_eval[-1]], y0, t_eval=t_eval).y.T
hnn_traj = hnn.trajectory([t_eval[0], t_eval[-1]], y0, t_eval)
nn_traj = nn.trajectory([t_eval[0], t_eval[-1]], y0, t_eval)

# Plotting
fig, axs = plt.subplots(2, 1, sharex=True)

axs[0].plot(t_eval, true_traj[:, 0], label=r"True $q$")
axs[0].plot(t_eval, hnn_traj[:, 0], "--", label=r"HNN $q$")
axs[0].plot(t_eval, nn_traj[:, 0], "-.", label=r"NN $q$")
axs[0].set_ylabel(r"Position $q$")
axs[0].legend()
axs[0].grid()

axs[1].plot(t_eval, true_traj[:, 1], label=r"True $p$")
axs[1].plot(t_eval, hnn_traj[:, 1], "--", label=r"HNN $p$")
axs[1].plot(t_eval, hnn_traj[:, 1], "-.", label=r"NN $p$")
axs[1].set_ylabel(r"Momentum $p$")
axs[1].legend()
axs[1].set_xlabel("Time [s]")
axs[1].grid()

plt.tight_layout()
plt.show()

# %%


# Compute Hamiltonian from model or true equation
def compute_hamiltonian(qp, model=None):
    q = qp[:, 0:1]
    p = qp[:, 1:2]
    if model:
        H = model.value(tf.convert_to_tensor(qp, dtype=tf.float32)).numpy().flatten()
        return H - H[0] + compute_hamiltonian(qp[0:1], model=None)[0]
    else:
        return 0.5 * np.square(p).flatten() + (1 - np.cos(q)).flatten()


H_true = compute_hamiltonian(true_traj)
H_hnn = compute_hamiltonian(hnn_traj, hnn)
H_nn = compute_hamiltonian(nn_traj)

plt.figure()
plt.plot(t_eval, H_true, label=r"True $\mathcal{H}$")
plt.plot(t_eval, H_hnn, "--", label=r"HNN $\mathcal{H}_\theta$")
plt.plot(t_eval, H_nn, "--", label=r"NN $\mathcal{H}$")
plt.ylabel("Hamiltonian")
plt.xlabel("Time")
plt.grid()
plt.legend()

# %%
