import tensorflow as tf
import numpy as np


def init(layers, bias=True, seed=1234, dtype=tf.float32):
    Ws, bs = [], []
    default_rng = np.random.default_rng(seed)
    for i in range(len(layers) - 1):
        W = xavier_init([layers[i], layers[i + 1]], rng=default_rng)
        b = tf.zeros([1, layers[i + 1]])
        Ws.append(tf.Variable(W, dtype=dtype, name=f"W_{i}"))
        bs.append(tf.Variable(b, dtype=dtype, trainable=bias, name=f"b_{i}"))
    return Ws, bs


def xavier_init(size, rng=np.random.default_rng()):
    in_dim = size[0]
    out_dim = size[1]
    xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
    return rng.normal(size=[in_dim, out_dim], scale=xavier_stddev)


class NeuralNetwork(tf.Module):
    def __init__(
        self,
        layers,
        bias=True,
        in_phi=tf.tanh,
        phi=tf.identity,
        output_dim=None,
        seed=1234,
        dtype=tf.float32,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.layers = layers
        self.bias = bias
        self.dtype = dtype
        self.seed = seed
        self.reinit()
        self.in_phi = in_phi
        self.phi = phi
        self.output_dim = output_dim

    @tf.function
    def __call__(self, input):
        num_layers = len(self.Ws)
        H = tf.cast(input, self.dtype)
        for layer in range(0, num_layers - 1):
            W = self.Ws[layer]
            b = self.bs[layer]
            H = self.in_phi(tf.add(tf.matmul(H, W), b))
        W = self.Ws[-1]
        b = self.bs[-1]
        output = self.phi(tf.add(tf.matmul(H, W), b))
        output = tf.transpose(output)
        if self.output_dim is not None:
            output = tf.reshape(output, (-1, self.output_dim[0], self.output_dim[1]))
        return output

    def reinit(self):
        self.Ws, self.bs = init(
            layers=self.layers, bias=self.bias, seed=self.seed, dtype=self.dtype
        )
