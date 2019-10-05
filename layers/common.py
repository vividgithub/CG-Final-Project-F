from utils.computil import ComputationContext
import tensorflow as tf
from utils.confutil import register_conf


@register_conf(name="data-split", scope="layer", conf_func="self")
class DataSplitLayer(tf.keras.layers.Layer):
    """
    Split the input data from (B, N, 3 + F) to a tuple with positions (B, N, 3) and (B, N, F)
    """
    def __init__(self, **kwargs):
        super(DataSplitLayer, self).__init__()

    def call(self, inputs, *args, **kwargs):
        return inputs[..., :3], inputs[..., 3:]


@register_conf(name=["output-segmentation", "output-classification"], scope="layer", conf_func="self")
class OutputClassificationSegmentationLayer(tf.keras.layers.Layer):
    """
    The final output layer, normally it is a dense layer for converting the feature dimension
    into the size of number of class for output
    """
    def __init__(self, class_count, use_position=False, **kwargs):
        """
        Initialization
        :param class_count: The number of class for output
        :param use_position: Whether to use the position as extra input for the dense layer
        """
        super(OutputClassificationSegmentationLayer, self).__init__()
        self.class_count = class_count
        self.use_posiiton = use_position
        self.dense = tf.keras.Dense(class_count)

    def call(self, inputs, *args, **kwargs):
        x = tf.concat(inputs, axis=-1) if self.use_posiiton else inputs[1]  # (B, N, F) or (B, N, F + 3)
        x = self.dense(x, *args, **kwargs)  # (B, N, class_count)
        return x

    def count_params(self):
        return self.dense.count_params()


@register_conf(name="output-conditional-segmentation", scope="layer", conf_func="self")
class OutputConditionalSegmentationLayer(tf.keras.layers.Layer):
    """
    An output layer for outputting the logits value. Inspired by the code from PointCNN, for a
    input (B, N, F), it uses a dense layer to map the feature exact to the number of classification, which
    is (B, N, F) -> (B, N, @class_count). In addition, in testing/validation stage, it first use a
    reduce mean operator to map (B, N, F) to (B, 1, F) for testing classification.
    """
    def __init__(self, class_count, use_position=False, **kwargs):
        """
        Initialize the layer
        :param class_count: The class count for the classification task
        :param use_position: Whether to use position as extra input to the dense layer
        """
        super(OutputConditionalSegmentationLayer, self).__init__()
        self.dense = tf.keras.layers.Dense(class_count)
        self.use_position = use_position

    def call(self, inputs, training, *args, **kwargs):
        x = tf.concat(inputs, axis=-1) if self.use_position else inputs[1]  # (B, N, F) or (B, N, F + 3)
        if not training:
            x = tf.reduce_mean(x, axis=1, keepdims=True)  # (B, N, F) for training and (B, 1, F) for testing
        x = self.dense(x, *args, **kwargs)  # (B, N/1, class_count)
        return x

    def count_params(self):
        return self.dense.count_params()


@register_conf(name="feature-reshape", scope="layer", conf_func="self")
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
        p = inputs[0]  # p: (B, N, 3)
        f = inputs[1]  # f: (B, N, F)
        f_ = c(self.layers, f)  # (B, N, F) -> (B, N, self.channels[-1])

        return p, f_
