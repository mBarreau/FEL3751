import tensorflow as tf
import tensorflow_probability as tfp
from tqdm import tqdm
from neural_network import NeuralNetwork


class NeuralODE:
    def __init__(self, layers, ss, weight_epsilon=0.1, seed=1234):
        self.weight_epsilon = weight_epsilon
        self.epsilon = NeuralNetwork([ss.n] + layers + [ss.n], seed=seed)
        self.f = ss.f
        self.g = ss.g
        self.h = ss.h
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)
        self.data = None
        self.x0 = tf.Variable(tf.ones((ss.n, 1)), dtype=tf.float32)
        self.ode_solver = tfp.math.ode.DormandPrince(atol=1e-6, rtol=1e-5)

    def set_data(self, data, u):
        self.data = (
            tf.convert_to_tensor(data[0], dtype=tf.float32),
            tf.convert_to_tensor(data[1], dtype=tf.float32),
        )
        self.u = u

    @tf.function
    def _call(self, t, x0, u):
        def dynamics(t, x):
            x = tf.reshape(x, (-1, 1))
            t = tf.reshape(t, (-1, 1))
            dxdt = (
                self.f(x)
                + self.g(x) @ u(t)
                + self.weight_epsilon * self.epsilon(tf.transpose(x))
            )
            return tf.reshape(dxdt, (-1,))

        result = self.ode_solver.solve(
            dynamics,
            initial_time=0.0,
            initial_state=tf.reshape(x0, (-1,)),
            solution_times=tf.reshape(t, (-1,)),
        )
        return tf.transpose(result.states)

    def __call__(self, t, x0=None, u=None):
        if x0 is None:
            x0 = self.x0
        if u is None:
            u = self.u
        return self._call(t, x0, u)

    def y(self, t, x0=None, u=None):
        return self.h(self(t, x0, u))

    def get_mse_data(self):
        return tf.reduce_mean(tf.square(self.data[1] - self.y(self.data[0])))

    @tf.function
    def update(self):
        vars = self.epsilon.trainable_variables + (self.x0,)
        with tf.GradientTape() as tape:
            loss = self.get_mse_data()
        grads = tape.gradient(loss, vars)
        self.optimizer.apply_gradients(zip(grads, vars))
        return loss

    def train(self, epochs=3000):
        losses = []
        pbar = tqdm(range(epochs))
        for i in pbar:
            loss = self.update()
            pbar.set_description(f"Loss: {loss.numpy():.6f}")
            losses.append(loss.numpy())
        return losses
