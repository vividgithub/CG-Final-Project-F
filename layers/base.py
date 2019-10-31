import tensorflow as tf
from utils.computil import ComputationContextLayer, get_regularizer_from_weight_decay


class ComposeLayer(tf.keras.layers.Layer):
    """
    A layer that uses other layer as sublayer for assistant computation
    """
    def __init__(self, *args, **kwargs):
        super(ComposeLayer, self).__init__(*args, **kwargs)
        self.sub_layers = dict()

    def count_params(self):
        return sum([l.count_params() for l in self.sub_layers.values()])

    def add_layer(self, label, layer_type, *args, **kwargs):
        """
        Add a layer as its sublayer for computation if the unique name of the layer if not exists
        :param label: An unique name of that layer
        :param layer_type: The class of the layer
        :param args: The args for initialization of the layer
        :param kwargs: The kwargs for initialization of the layer
        :return: An instance of that layer
        """
        if self.sub_layers.get(label) is None:
            self.sub_layers[label] = ComputationContextLayer(layer_type(*args, **kwargs))
        return self.sub_layers[label]

    def dense(self, name, units, activation, weight_decay):
        """
        A simple alias to create a Dense layer in compose layer
        :param name: The name of the layer
        :param units: The number of output features of the final feature dimensions
        :param activation: The activation
        :param weight_decay: The weight decay, a float value for l2 weight decay, or a two tuple for l1,l2 weight decay
        :return: A dense layer
        """
        return self.add_layer(
            name,
            tf.keras.layers.Dense,
            units=units,
            activation=activation,
            kernel_regularizer=get_regularizer_from_weight_decay(weight_decay),
            bias_regularizer=get_regularizer_from_weight_decay(weight_decay),
            name=name
        )

    def batch_normalization(self, name, momentum, weight_decay):
        """
        Add a batch normalization layer in the compose layer
        :param name: The name of the layer
        :param momentum: The momentum of the batch normalization
        :param weight_decay: The weight decay for the layer
        :return: A batch normalization layer
        """
        return self.add_layer(
            name,
            tf.keras.layers.BatchNormalization,
            momentum=momentum,
            beta_regularizer=get_regularizer_from_weight_decay(weight_decay),
            gamma_regularizer=get_regularizer_from_weight_decay(weight_decay),
            name=name
        )

    def activation_(self, name, activation):
        """
        Add a activation layer in the compose layer
        :param name: The name of the layer
        :param activation: The activation of the layer
        :return: A activation layer
        """
        return self.add_layer(
            name,
            tf.keras.layers.Activation,
            activation=activation,
            name=name
        )

    def unary_conv(self, name, channel, activation, momentum, weight_decay):
        """
        Adding a unary convolution layer (in KPConv, Dense --> Normalization --> Activation).
        :param name: The name of the unary convolution
        :param channel: The output channel for the unary convolution
        :param activation: The activation, None for not using any activation
        :param momentum: The momentum used in batch normalization, none for not using any batch normalization
        :param weight_decay: The weight decay for the unary convolution
        :return: The unary convolution layer
        """
        ls = [self.dense(name + "-Unary", channel, activation=None, weight_decay=weight_decay)]
        if momentum is not None:
            ls.append(self.batch_normalization(name + "-Normalization", momentum=momentum, weight_decay=weight_decay))
        if activation is not None:
            ls.append(self.activation_(name, activation=activation))

        return tuple(ls)