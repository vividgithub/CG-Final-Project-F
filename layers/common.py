from computil import ComputationContext
import tensorflow as tf

class FeatureReshapeLayer(tf.keras.layers.Layer):
    """
    A feature resize layer that reshape the feature dimension using several dense layer with dropout
    """
    def __init__(self, channels, dropout, activation="elu", momentum=0.99, **kwargs):
        super(FeatureReshapeLayer, self).__init__()

        self.channels = channels
        self.dropout = dropout
        self.layers = []

        for channel_size, dropout_rate in zip(channels, dropout):
            self.layers.append(tf.keras.layers.Dense(channel_size, activation="elu"))
            self.layers.append(tf.keras.layers.BatchNormalization(momentum=momentum))
            if dropout_rate > 0.0:
                self.layers.append(tf.keras.layers.Dropout(dropout_rate))

    def count_params(self):
        return sum([layer.count_params() for layer in self.layers])

    def call(self, inputs, *args, **kwargs):
        c = ComputationContext(*args, **kwargs)
        p = inputs[:, :, :3]  # (B, N, 3)
        f = inputs[:, :, 3:]  # (B, N, F)

        f_ = c(self.layers, f)  # (B, N, F) -> (B, N, self.channels[-1])

        return tf.concat([p, f_], axis=-1)