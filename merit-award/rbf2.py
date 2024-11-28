import keras
from keras import layers
import tensorflow as tf
import numpy as np

class RBFLayer(layers.Layer):
    def __init__(self, centers, widths, **kwargs):
        super(RBFLayer, self).__init__(**kwargs)
        self.centers = tf.convert_to_tensor(centers, dtype=tf.float32)
        self.widths = tf.convert_to_tensor(widths, dtype=tf.float32)

    def build(self, input_shape):
        # This is where we would initialize any learnable parameters if needed
        pass

    def call(self, inputs):
        # Compute the RBF activations
        # Inputs is the input data passed to the layer
        expanded_input = tf.expand_dims(inputs, 1)  # Shape: (batch_size, 1, input_dim)
        expanded_centers = tf.expand_dims(self.centers, 0)  # Shape: (1, num_centers, input_dim)
        
        # Compute the squared Euclidean distance
        dist = tf.reduce_sum(tf.square(expanded_input - expanded_centers), axis=-1)  # Shape: (batch_size, num_centers)
        
        # Compute RBF activations
        rbf_output = tf.exp(-dist / (2 * self.widths ** 2))
        return rbf_output

# Example usage: creating an RBF network model
def create_rbf_network(input_dim, centers, widths, output_dim):
    inputs = layers.Input(shape=(input_dim,))
    rbf = RBFLayer(centers=centers, widths=widths)(inputs)
    outputs = layers.Dense(output_dim)(rbf)  # Dense layer for output
    model = keras.models.Model(inputs, outputs)
    return model