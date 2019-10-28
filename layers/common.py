from utils.computil import ComputationContext, get_regularizer_from_weight_decay
import tensorflow as tf
from utils.confutil import register_conf
from layers.base import ComposeLayer


@register_conf(name="input-feature-extend", scope="layer", conf_func="self")
class InputFeatureExtendLayer(tf.keras.layers.Layer):
    """
    This layer is used to extend additional point features. For example, extend one-hot 1 to each feature, or
    by passing the global position to the point features. We currently accept several methods: "none" for not extending
    the feature, "one" for placing adding a one-hot 1s to the feature dimensions. And "pos" to extend the input
    positions to the features
    """
    def __init__(self, method, label=None):
        """
        Initialization
        :param method: The method to extend the feature, accept "none", "one", "pos"
        :param label: An optional label for this layer
        """
        super(InputFeatureExtendLayer, self).__init__(name=label)
        self.method = method

    def call(self, inputs, *args, **kwargs):
        # points: (B, (N), 3)
        # features: (B, (N), F)
        points, features = inputs[0], inputs[1]

        if self.method == "none":
            pass
        elif self.method == "one":
            one_features = tf.ones_like(points)[..., :1]  # (B, (N), 1)
            features = tf.concat([features, one_features], axis=-1)
        elif self.method == "pos":
            features = tf.concat([features, points], axis=-1)
        else:
            assert False, f"The method \"{self.method}\" is not supported yet"

        return points, features


@register_conf(name="data-split", scope="layer", conf_func="self")
class DataSplitLayer(tf.keras.layers.Layer):
    """
    Split the input data from (B, N, 3 + F) to a tuple with positions (B, N, 3) and (B, N, F)
    """
    def __init__(self, label=None, **kwargs):
        """
        Initialization
        :param label: An optional label for the layer
        """
        super(DataSplitLayer, self).__init__(name=label)

    def call(self, inputs, *args, **kwargs):
        return inputs[..., :3], inputs[..., 3:]


@register_conf(name=["output-segmentation", "output-classification"], scope="layer", conf_func="self")
class OutputClassificationSegmentationLayer(tf.keras.layers.Layer):
    """
    The final output layer, normally it is a dense layer for converting the feature dimension
    into the size of number of class for output
    """
    def __init__(self, class_count, use_position=False, weight_decay=0.0, label=None, **kwargs):
        """
        Initialization
        :param class_count: The number of class for output
        :param use_position: Whether to use the position as extra input for the dense layer
        :param weight_decay: The weight decay used in this layer
        :param label: An optional label for the layer
        """
        super(OutputClassificationSegmentationLayer, self).__init__(name=label)
        self.class_count = class_count
        self.use_posiiton = use_position
        self.dense = tf.keras.layers.Dense(
            class_count,
            kernel_regularizer=get_regularizer_from_weight_decay(weight_decay),
            bias_regularizer=get_regularizer_from_weight_decay(weight_decay)
        )

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
    def __init__(self, class_count, use_position=False, weight_decay=0.0, label=None, **kwargs):
        """
        Initialize the layer
        :param class_count: The class count for the classification task
        :param use_position: Whether to use position as extra input to the dense layer
        :param weight_decay: The weight decay for this layer
        :param label: An optional label for the layer
        """
        super(OutputConditionalSegmentationLayer, self).__init__(name=label)
        self.use_position = use_position
        self.dense = tf.keras.layers.Dense(
            class_count,
            kernel_regularizer=get_regularizer_from_weight_decay(weight_decay),
            bias_regularizer=get_regularizer_from_weight_decay(weight_decay)
        )

    def call(self, inputs, training, *args, **kwargs):
        x = tf.concat(inputs, axis=-1) if self.use_position else inputs[1]  # (B, N, F) or (B, N, F + 3)
        if not training:
            x = tf.reduce_mean(x, axis=1, keepdims=True)  # (B, N, F) for training and (B, 1, F) for testing
        x = self.dense(x, *args, **kwargs)  # (B, N/1, class_count)
        return x

    def count_params(self):
        return self.dense.count_params()


@register_conf(name="feature-reshape", scope="layer", conf_func="self")
class FeatureReshapeLayer(ComposeLayer):
    """
    A feature resize layer that reshape the feature dimension using several dense layer with dropout
    """
    def __init__(self, channels, dropout, activation="elu", momentum=0.99, bn_first=True,
                 weight_decay=0.0, label=None, **kwargs):
        """
        Initialization
        :param channels: A list defines the number of features for output for each dense layer
        :param dropout: A list defines the dropout rate for each dense layer
        :param activation: The activation function for each dense layer
        :param momentum: The momentum for batch normalization after each dense layer
        :param bn_first: Whether to use batch normalization before activation or not
        :param weight_decay: The weight decay for this layer
        :param label: An optional label for the layer
        """
        super(FeatureReshapeLayer, self).__init__(name=label)
        self.channels = channels
        self.dropout = dropout
        self.activation = activation
        self.momentum = momentum
        self.bn_first = bn_first
        self.weight_decay = weight_decay

    def call(self, inputs, *args, **kwargs):
        c = ComputationContext(*args, **kwargs)
        p = inputs[0]  # p: (B, N, 3)
        f = inputs[1]  # f: (B, N, F)

        for i, (channel_size, dropout_rate) in enumerate(zip(self.channels, self.dropout)):
            dense = self.dense("Dense", channel_size, None, weight_decay=self.weight_decay)
            bn = self.batch_normalization("BatchNormalization", momentum=self.momentum, weight_decay=self.weight_decay)
            activation_ = self.activation_("Activation", activation=self.activation)

            ls = (dense, bn, activation_) if self.bn_first else (dense, activation_, bn)

            f = c(ls, f, name=f"Level-{i}")

        return p, f
