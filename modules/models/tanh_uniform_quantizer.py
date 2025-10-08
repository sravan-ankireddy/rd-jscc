import tensorflow as tf

class NBitQuantizationLayer(tf.keras.layers.Layer):
    def __init__(self, n_bits, **kwargs):
        super(NBitQuantizationLayer, self).__init__(**kwargs)
        self.n_bits = n_bits
        self.num_levels = 2 ** n_bits  # Number of quantization levels

    def build(self, input_shape):
        pass  # No trainable parameters needed for this layer

    def call(self, inputs):
        # Apply the tanh activation function
        x = tf.tanh(inputs)

        # Scale the tanh output from [-1, 1] to [0, 1]
        x_scaled = (x + 1.0) / 2.0

        # Perform quantization
        x_quantized = tf.round(x_scaled * (self.num_levels - 1)) / (self.num_levels - 1)

        # Rescale quantized values back to the original range of tanh
        x_rescaled = 2.0 * x_quantized - 1.0

        # Use stop_gradient to ensure gradients flow through the original x (for straight-through estimator)
        x_straight_through = x + tf.stop_gradient(x_rescaled - x)

        return x_straight_through