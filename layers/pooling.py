import tensorflow as tf
from utils.confutil import register_conf


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
        if method == "average":
            self.reduce_op = tf.reduce_mean
        elif method == "max":
            self.reduce_op = tf.reduce_max
        else:
            assert False, f"Method \"{method}\" in Pooling layer is not supported"

    def call(self, inputs, *args, **kwargs):
        # points: (N, 3)
        # features: (N, F)
        # neighbor_indices: (N', (neighbor))

        # Get input
        points, features, neighbor_indices = inputs[0], inputs[1], inputs[2]

        # Gather and reduce
        neighbor_features = tf.gather(features, neighbor_indices)  # (N, (neighbor), F)
        output_features = self.reduce_op(neighbor_features, axis=1)

        return output_features
