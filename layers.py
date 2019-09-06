import tensorflow as tf
import legacy.pointfly as pf
from computil import ComputationContext


class XConvLayerCoreV2(tf.keras.layers.Layer):
    """
    The X-Conv kernel used in "PointCNN"(https://arxiv.org/abs/1801.07791)
    """
    def __init__(self, p, k, d, c, cpf, depth_multiplier=2, sampling="random", sorting_method=None, with_global=False):
        """
        The X-Conv kernel, from "PointCNN"(https://arxiv.org/abs/1801.07791). It takes an tensor input
        (BxNxF) and generate a tensor with shape (Bxpxc).
        :param p: The number of points for output, if p < 0, then it uses input point size
        :param k: The number of neighborhood to convolve
        :param d: The dilation factor. The kernel will search kxd neighbor but only remains d points for convolution
        :param c: The channel for output
        :param cpf: The \f$C_{\sigma}\f$ in the paper, which indicates the channel number for converting position into
        feature
        :param depth_multiplier: The depth multiplier used in separable convolution
        :param sampling: How to sampling the output point locations, "random" will sample randomly; "fps" will use
        farthest neighbor sampling method
        :param sorting_method: How to give the point order before convolution. Currently it is just a placeholder
        :param with_global: Whether to add the global position in convolution
        """
        super(XConvLayerCoreV2, self).__init__()
        self.p = p
        self.k = k
        self.d = d
        self.c = c
        self.cpf = cpf
        self.depth_multiplier = depth_multiplier
        assert sampling == "random", "Fps and other sampling method is not supported right now"
        self.sampling = sampling
        self.sorting_method = sorting_method
        self.with_global = with_global

        # l_pos_to_feature doing a series operation to expand position (B, N, 3) into large tensor space (B, N, cpf)
        self.l_pos2feature = (
            tf.keras.layers.Dense(self.cpf),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(self.cpf),
            tf.keras.layers.BatchNormalization()
        )

        # The X-conv core, it uses 3 convolution kernel to transform the local coordinate (B, p, k, 3)
        # ->(conv1, bn) -> (B, p, k, k) ->(conv2, bn) -> (B, p, k, k) ->(conv3, bn) -> (B, p, k, k)
        self.l_xconv1 = (
            tf.keras.layers.Conv2D(self.k * self.k, (1, self.k), activation="elu"),
            tf.keras.layers.BatchNormalization()
        )
        self.l_xconv2 = (
            tf.keras.layers.DepthwiseConv2D((1, self.k), depth_multiplier=self.k, activation="elu"),
            tf.keras.layers.BatchNormalization()
        )
        self.l_xconv3 = (
            tf.keras.layers.DepthwiseConv2D((1, self.k), depth_multiplier=self.k, activation="elu"),
            tf.keras.layers.BatchNormalization()
        )

        # Final convolution, converts (B, p, k, cpf + F) to (B, p, c)
        self.l_final_conv = (
            tf.keras.layers.SeparableConv2D(self.c, (1, self.k), depth_multiplier=self.depth_multiplier,
                                            activation="elu"),
            tf.keras.layers.BatchNormalization()
        )

        # Convert the global position to feature
        self.l_global_pos2feature = (
            tf.keras.layers.Dense(self.c // 4, activation="elu"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(self.c // 4, activation="elu"),
            tf.keras.layers.BatchNormalization()
        )

    def call(self, inputs, *args, **kwargs):
        c = ComputationContext(*args, **kwargs)
        p = inputs[:, :, :3]  # (B, N, 3)
        f = inputs[:, :, 3:]  # (B, N, F)

        shape = tf.shape(inputs)
        B = shape[0]
        N = self.p if self.p > 0 else shape[1]

        # Get the sampling points
        # Not so randomly, only follow the original implementations
        q = p if self.p < 0 else p[:, :self.p, :]  # (B, p, 3)

        # Get the neighborhood indices
        _, indices_dilated = pf.knn_indices_general(q, p, self.k * self.d, True)  # (B, p, k*d, 2)
        indices = indices_dilated[:, :, ::self.d, :]  # (B, p, k, 2)

        # Sort the points
        # TODO:

        # Group and convert to local coordinate
        p_ = tf.gather_nd(p, indices) - tf.expand_dims(q, axis=2)  # (B, p, k, 3) - (B, p, 1, 3) = (B, p, k, 3)

        # Convert position into features
        f_ = c(self.l_pos2feature, p_)  # (B, p, k, 3) -> (B, p, k, cpf)

        # Concat the origin feature
        f_ = tf.concat([f_, tf.gather_nd(f, indices)], axis=-1)  # (B, p, k, cpf) ~concat~ (B, p, k, F) = (B, p, k, cpf + F)

        # X convolution core
        x_conv_mat = tf.reshape(c(self.l_xconv1, p_), (B, N, self.k, self.k))  # (B, p, k, 3) -> (B, p, k, k)
        x_conv_mat = tf.reshape(c(self.l_xconv2, x_conv_mat), (B, N, self.k, self.k))  # (B, p, k, k) -> (B, p, k, k)
        x_conv_mat = tf.reshape(c(self.l_xconv3, x_conv_mat), (B, N, self.k, self.k))  # (B, p, k, k) -> (B, p, k, k)

        # Matrix multiplication
        f_ = tf.linalg.matmul(x_conv_mat, f_)  # (B, p, k, k) x (B, p, k, cpf + f) = (B, p, k, cpf + f)

        # Final convolution
        f_ = tf.squeeze(c(self.l_final_conv, f_), axis=2)  # (B, p, k, cpf + f) -> (B, p, 1, c) ->(squeeze)-> (B, p, c)

        # With global feature
        if self.l_global_pos2feature:
            f_global = c(self.l_global_pos2feature, q)  # (B, p, 3) -> (B, p, c/4)
            f_ = tf.concat([f_global, f_], axis=-1)  # (B, p, c/4) ~concat~ (B, p, c) = (B, p, c + c/4)

        return f_


XConvPoolingLayer = XConvLayerCoreV2

class XConvLayer(XConvLayerCoreV2):
    """
    The actual convolution layer. Different from the XConvLayerCore, it doesn't take "p" as a parameter, it maps
    an input of (BxNxF) tensor to (BxNxc) tensor.
    """
    def __init__(self, *args, **kwargs):
        super(XConvLayer, self).__init__(-1, *args, **kwargs)