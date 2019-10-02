from computil import ComputationContext
import tensorflow as tf


class OutputConditionalSegmentationLayer(tf.keras.layers.Layer):
    """
    An output layer for outputting the logits value. Inspired by the code from PointCNN, for a
    input (B, N, F), it uses a dense layer to map the feature exact to the number of classification, which
    is (B, N, F) -> (B, N, @class_count). In addition, in testing/validation stage, it first use a
    reduce mean operator to map (B, N, F) to (B, 1, F) for testing classification.
    """
    def __init__(self, class_count, **kwargs):
        """
        Initialize the layer
        :param class_count: The class count for the classification task
        """
        super(OutputConditionalSegmentationLayer, self).__init__()
        self.dense = tf.keras.layers.Dense(class_count)

    def call(self, inputs, training, *args, **kwargs):
        x = inputs if training else tf.reduce_mean(inputs, axis=1, keepdims=True)  # (B, N/1, F)
        x = self.dense(x, *args, **kwargs)  # (B, N/1, @class_count)
        return x

    def count_params(self):
        return self.dense.count_params()


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
            self.layers.append(tf.keras.layers.Dense(channel_size, activation=activation))
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