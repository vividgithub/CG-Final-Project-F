import tensorflow as tf
from utils.confutil import register_conf


def _get_pooling_op_from_name(name):
    if name == "average" or name == "mean":
        return tf.reduce_mean
    elif name == "max":
        return tf.reduce_max
    else:
        assert False, f"Method \"{name}\" is not a correct pooling configuration"


@register_conf("pooling-pooling", scope="layer", conf_func="self")
class PoolingLayer(tf.keras.layers.Layer):
    """
    Point pooling layer, it accepts a stacked points: (N, 3), stacked features: (N, F),
    the neighbor information (N', (neighbor)) and output the pooling feature (N', F)
    """
    def __init__(self, method, label=None, **kwargs):
        """
        Initialization
        :param method: The pooling method, "average" for average pooling, and "max" for max-pooling
        :param label: A optional label for this layer
        """
        super(PoolingLayer, self).__init__(name=label)
        self.method = method
        self.reduce_op = _get_pooling_op_from_name(method)

    def call(self, inputs, *args, **kwargs):
        # points: (N, 3)
        # features: (N, F)
        # neighbor_indices: (N', (neighbor))

        # Get input
        points, features, neighbor_indices = inputs[0], inputs[1], inputs[2]

        # Gather and reduce
        neighbor_features = tf.gather(features, neighbor_indices)  # (N', (neighbor), F)
        output_features = self.reduce_op(neighbor_features, axis=1)

        return output_features


@register_conf("pooling-global", scope="layer", conf_func="self")
class GlobalPoolingLayer(tf.keras.layers.Layer):
    """
    Global pooling layer, it can be used individually. It accepts a RaggedTensor points: (B, (N), 3) and RaggedTensor
    features: (B, (N), F). It reduce the features to (B, F) with average or max pooling and discards the points feature.
    """
    def __init__(self, method, label=None, **kwargs):
        """
        Initialization
        :param method: The pooling method name, "average" or "max"
        :param label: The optional label for this layer
        """
        super(GlobalPoolingLayer, self).__init__(name=label)
        self.method = method
        self.reduce_op = _get_pooling_op_from_name(method)

    def call(self, inputs, *args, **kwargs):
        # points: (B, (N), 3)
        # features: (B, (N), F)
        # output_features: (B, F)
        points, features = inputs[0], inputs[1]
        return points, self.reduce_op(features, axis=1)
