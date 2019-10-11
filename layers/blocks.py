import tensorflow as tf
from utils.confutil import register_conf, object_from_conf
from utils.computil import ComputationContext


def unpack_ragged(x, name=None):
    """
    Accept a Tensor (B, N, x) or a ragged tensor (B, (N), x) and try to stack it by deleting the first dimension and
    returns (N', x), where N' is the total N in all batches.
    :param x: An input tensor Tensor or ragged Tensor
    :param name: An optional name for the op
    :return: Two value "x'" and "row_splits", where x' is the stacked output and "row_splits" is the same as the
    "row_splits" in RaggedTensor.
    """
    with tf.name_scope(name or "UnpackRagged"):
        if isinstance(x, tf.RaggedTensor):
            # We don't use tf.is_ragged since it is unopened
            return x.values, x.row_splits

        assert tf.is_tensor(x), f"Unexpected type type(x)={type(x)}, expected: Tensor or RaggedTensor"

        # Handle tensor
        x_shape = tf.shape(x)
        b, n = x_shape[0], x_shape[1]
        x_ = tf.reshape(x, (b * n, -1))
        row_splits = tf.range(0, b * n + 1, n)
        return x_, row_splits


# @register_conf("block-simple", scope="layer")
# class SimpleBlock(tf.keras.layers.Layer):
#     """
#     A simple convolution block, it accepts an input of (points, features), which points is (B, (N), 3) and features
#     is (B, (N), F). It uses custom convolution op to generate new features, new_features: (B, (N), F') and output
#     (points, new_features).
#     """
#     def __init__(self, neighbor_layer, conv_layer, label=None, **kwargs):
#         """
#         Initialization
#         :param neighbor_layer: The neighbor layer accepts an inputs (N, 3) and (N', 3) and their row splits value
#         and generate a tuple representing ragged tensor for neighbor indices (N', (neighbor))
#         :param conv_layer: The convolution layer accepts points: (N, 3), features: (N, F), output_points: (N', 3) and
#         the neighbor indices: (N', (neighbor)) and generate the new point feature (N', F')
#         :param label: An optional label for the layer
#         and generate the convolution result new_features: (N', F'). In simple block, output_points = points
#         """
#         super(SimpleBlock, self).__init__(name=label)
#         self.neighbor_layer = neighbor_layer
#         self.conv_layer = conv_layer
#
#     @staticmethod
#     def conf_func(conf):
#         """
#         Defines how to convert a configuration dict to a SimpleBlock object
#         :param conf: The configuration dict
#         :return: SimpleBlock object
#         """
#         return SimpleBlock(
#             neighbor_layer=lambda: object_from_conf(conf["neighbor"], scope="layer"),
#             conv_layer=lambda: object_from_conf(conf["conv"], scope="layer"),
#             **{k: v for k, v in conf.items() if k != "neighbor" and k != "conv"}  # By pass others
#         )
#
#     def count_params(self):
#         params_neighbor_op = self.neighbor_layer.count_params() if hasattr(self.neighbor_layer, "count_params") else 0
#         params_conv_op = self.conv_layer.count_params() if hasattr(self.conv_layer, "count_params") else 0
#         return params_conv_op + params_neighbor_op
#
#     def call(self, inputs, *args, **kwargs):
#         points, features = inputs[0], inputs[1]
#
#         # Stacked the points and features by deleting the batch dimensions
#         stacked_points, points_splits = unpack_ragged(points, name="UnpackPoints")
#         stacked_features, features_splits = unpack_ragged(features, name="UnpackFeatures")
#
#         # Get the neighbor indices
#         neighbor_indices = self.neighbor_layer(stacked_points, stacked_points,
#                                                points_splits, points_splits, *args, **kwargs)
#
#         # Convolution
#         with tf.name_scope("Convolution"):
#             stacked_output_features = self.conv_layer(stacked_points, stacked_features,
#                                                       stacked_points, neighbor_indices, *args, **kwargs)
#
#             output_features = tf.RaggedTensor(stacked_output_features, features_splits)
#
#         return points, output_features


@register_conf("block-res", scope="layer")
class ResBlock(tf.keras.layers.Layer):

    _SIMPLE = "simple"
    _NORMAL = "normal"
    _BOTTLENECK = "bottleneck"
    _BOTTLENECK_STRIDED = "bottleneck-strided"

    @staticmethod
    def SHOULD_SAMPLING(structure):
        return structure != ResBlock._BOTTLENECK_STRIDED

    @staticmethod
    def HAS_SHORTCUT(structure):
        return structure != ResBlock._SIMPLE

    @staticmethod
    def BOTTLENECK(structure):
        return structure in [ResBlock._BOTTLENECK, ResBlock._BOTTLENECK_STRIDED]

    """
    Resnet-like block for 3D point classification and segmentation. Different from the simple block, it adds an
    shortcut link and sums the final value to the output, just like the residual block that used in image.
    Currently it supports 3 types of structures. The "simple" structure only contains the mainline without the shortcut. 
    The mainline consists of (Conv -> Activation -> BatchNormalization); the "normal" structure extended the "simple" 
    by adding its value with the shortcut. The "bottleneck" block, in which the mainline is constructed with
    (Dense -> Conv -> Dense), where the Dense is used to shrink or extend the last feature layer.
    The "bottleneck-strided" resembles the structure of "bottleneck", except that it does a sampling in the mainline
    and shortcut link, so the output will has smaller point set than the input
    """
    def __init__(self, structure, channel, neighbor_layer, conv_layer,
                 sampling_layer=None, pooling_layer=None, activation=None):
        """
        Initialization
        :param structure: The structure of layer, supports "simple", "normal", "bottleneck" and "bottleneck-strided"
        :param channel: The channel for output
        :param neighbor_layer: The layer to query neighbor. The neighbor layer accepts an inputs (N, 3) and (N', 3)
        and their row splits value and generate a tuple representing ragged tensor for neighbor indices (N', (neighbor))
        :param conv_layer: The convolution layer accepts points: (N, 3), features: (N, F), output_points: (N', 3) and
        the neighbor indices: (N', (neighbor)) and generate new point feature (N', F')
        :param sampling_layer: The sampling layer accepts points: (N, 3) and row_splits: (B + 1) and generate a
        sub-sampling points output_points: (N', 3) and output_row_splits: (B + 1). Sampling layer is only required
        for strided structure
        :param pooling_layer: The pooling layer accepts points: (N, 3), features: (N, F), output_points: (N', 3) and
        the neighbor indices: (N', (neighbor)) and generate pooled feature (N', F). Pooling layer is only required
        for strided structure
        :param activation: The activation used in the block
        """
        assert structure in [ResBlock._NORMAL, ResBlock._BOTTLENECK, ResBlock._BOTTLENECK_STRIDED, ResBlock._SIMPLE], \
            f"Structure {structure} is not supported in ResBlock"
        assert not (structure == ResBlock._BOTTLENECK_STRIDED and (sampling_layer is None or pooling_layer is None)), \
            f"Structure is {structure} but sampling layer or pooling layer is None, " \
            f"sampling_layer={sampling_layer}, pooling_layer={pooling_layer}"

        self.structure = structure
        self.channel = channel
        self.neighbor_layer = neighbor_layer
        self.conv_layer = conv_layer
        self.sampling_layer = sampling_layer
        self.pooling_layer = pooling_layer
        self.activation = activation or "leaky_relu"
        self.activation_func = tf.keras.activations.get(self.activation)

    def should_sampling(self):
        return self.SHOULD_SAMPLING(self.structure)

    def has_shortcut(self):
        return self.HAS_SHORTCUT(self.structure)

    def bottleneck(self):
        return self.BOTTLENECK(self.structure)

    @staticmethod
    def conf_func(conf):
        """
        Defines how to convert a configuration dict to a ResBlock object
        :param conf: The configuration dict
        :return: ResBlock object
        """
        channel = conf["channel"]
        structure = conf["structure"]
        channel_for_conv = channel // 4 if ResBlock.BOTTLENECK(structure) else channel

        # Parse the "neighbor", "conv", "sampling" and "pooling" to construct layer
        # and inject the channel context to the convolution layer
        return ResBlock(
            neighbor_layer=object_from_conf(conf["neighbor"], scope="layer"),
            conv_layer=object_from_conf(
                {channel: channel_for_conv, **conf["conv"]},
                scope="layer"
            ),
            sampling_layer=object_from_conf(conf["sampling"], scope="layer") if "sampling" in conf else None,
            pooling_layer=object_from_conf(conf["pooling"], scope="layer") if "pooling" in conf else None,
            **{k: v for k, v in conf.items() if k not in ["neighbor", "conv", "sampling", "pooling"]}
        )

    def count_params(self):
        layers = [self.neighbor_layer, self.conv_layer, self.sampling_layer, self.pooling_layer]
        return sum([
            layer.count_params() if hasattr(layer, "count_params") else 0 for layer in layers
        ])

    def call(self, inputs, *args, **kwargs):
        # points: (B, (N), 3)
        # features: (B, (N), F)
        points, features = inputs[0], inputs[1]

        # Stack the inputs
        points, row_splits = unpack_ragged(points, name="UnpackPoints")
        features, _ = unpack_ragged(features, name="UnpackFeatures")

        def unary_conv(x, fdim, activated, normalized):
            x = tf.keras.layers.Dense(fdim, activation=None)(x, *args, **kwargs)
            x = tf.keras.layers.BatchNormalization()(x, *args, **kwargs) if normalized else x
            x = self.activation_func(x) if activated else x
            return x

        # Determine the output_points (N', 3)
        # For convolution, it should equal to the points.
        # For strided-based structure, it will use sampling to get the output
        # output_points: (N', 3)
        # output_row_splits: (N', 3)
        if not self.should_sampling():
            output_points, output_row_splits = points, row_splits
        else:
            output_points, output_row_splits = self.sampling_layer(points, row_splits, *args, **kwargs)

        # Now we can get the neighbor information
        neighbor_indices = self.neighbor_layer(points, row_splits, output_points, output_row_splits, *args, **kwargs)

        # Shortcut line
        shortcut = None
        if self.has_shortcut():
            with tf.name_scope("Shortcut"):
                if self.should_sampling():
                    shortcut = self.pooling_layer(points, features, neighbor_indices, *args, **kwargs)  # (N', F)
                shortcut = unary_conv(shortcut, self.channel, activated=False, normalized=True)  # (N', F')

        # Mainline
        with tf.name_scope("Mainline"):
            # 1x1 convolution for bottleneck
            mainline = unary_conv(features, self.channel // 4, activated=True, normalized=True) \
                if self.bottleneck() else features  # (N', F) for non-bottleneck or (N', F'/4)
            # Convolution
            mainline = self.conv_layer(mainline, *args, **kwargs)  # (N', F') for non-bottleneck or (N', F'/4)
            # 1x1 convolution for bottleneck
            mainline = unary_conv(features, self.channel, activated=False, normalized=True) \
                if self.bottleneck() else mainline  # (N', F')

        # Add mainline and shortcut
        output_features = mainline + shortcut if self.has_shortcut() else mainline  # (N', F')
        # Final activation
        output_features = self.activation_func(output_features)

        return (
            tf.RaggedTensor(output_points, row_splits=output_row_splits),
            tf.RaggedTensor(output_features, row_splits=output_row_splits)
        )
