import tensorflow as tf
import numpy as np
from tqdm import tqdm

from utils import NeuralNetwork


class PINN:
    def __init__(self, layers, ss, N_phys=10, N_dual=10, T=5, seed=1234):
        self.x_hat = NeuralNetwork([1] + layers + [ss.n], seed=seed)
        self.n = ss.n
        self.optimizer_primal = tf.keras.optimizers.Adam(learning_rate=1e-3)
        self.optimizer_dual = tf.keras.optimizers.Adam(learning_rate=1e-3)
        self.weight = tf.Variable(0.0, dtype=self.x_hat.dtype)
        self.N_dual = N_dual
        self.N_phys = N_phys
        self.T = T
        self.f = ss.f
        self.g = ss.g
        self.h = ss.h
        self.data = None
        self.resample()

    def set_data(self, data, u):
        self.data = data[0], data[1]
        self.u = u

    def resample(self):
        t_tf = tf.convert_to_tensor(
            np.random.rand(int(self.N_phys * self.T), 1) * self.T
        )
        self.t_tf = tf.cast(t_tf, self.x_hat.dtype)

    def __call__(self, t):
        return self.x_hat(tf.transpose(t))

    def y(self, t):
        return self.h(self(t))

    def get_residual(self, t_tf):
        dx_hat_tf = []
        t = tf.transpose(t_tf)
        for i in range(self.n):
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(t)
                x_hat_tf = self(t)[i]
            grads = tape.gradient(x_hat_tf, t)
            dx_hat_tf.append(tf.reshape(grads, (1, -1)))
        dx_hat_tf = tf.concat(dx_hat_tf, 0)
        residual = None
        return residual

    def get_mse_data(self):
        return 0.0

    def get_mse_residual(self):
        return 0.0

    # @tf.function
    def get_cost(self):
        return self.get_mse_data() + self.weight * self.get_mse_residual()

    # @tf.function
    def primal_update(self):
        pass

    # @tf.function
    def dual_update(self):
        pass

    def train(self, epochs=3000):
        losses = []
        weights = []
        self.resample()
        pbar = tqdm(range(epochs))
        for i in pbar:
            self.primal_update()
            if i % self.N_dual == 0 and i > 0:
                self.dual_update()
                self.resample()
            loss = self.get_cost().numpy()
            pbar.set_description(f"Loss: {loss:.6f}")
            losses.append(loss)
            weights.append(self.weight.numpy())
        return losses, weights
