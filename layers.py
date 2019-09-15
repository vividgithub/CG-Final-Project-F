import tensorflow as tf
import legacy.pointfly as pf
from computil import ComputationContext


class XConvLayerCoreV1(tf.keras.layers.Layer):
    """
    The X-Conv kernel used in "PointCNN"(https://arxiv.org/abs/1801.07791)
    """
    def __init__(self, p, k, d, c, cpf, depth_multiplier=2, sampling="random", sorting_method=None, with_global=False, **kwargs):
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
        super(XConvLayerCoreV1, self).__init__()
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
        self.sub_layers = dict()

    def batch_normalization(self, inputs, training, name):
        layer = self.sub_layers.get(name)
        if not layer:
            layer = tf.keras.layers.BatchNormalization(momentum=0.99, name=name)
            self.sub_layers[name] = layer
        return layer(inputs, training=training)

    def dense(self, inputs, units, name, training):
        layer = self.sub_layers.get(name)
        if not layer:
            layer = tf.keras.layers.Dense(units, use_bias=False, name=name)
            self.sub_layers[name] = layer

        x = layer(inputs, training=training)
        x = self.batch_normalization(x, training=training, name=name+"-BN")
        return x

    def conv2d(self, inputs, channel, name, training, kernel_size):
        layer = self.sub_layers.get(name)
        if not layer:
            layer = tf.keras.layers.Conv2D(channel, kernel_size, use_bias=False, activation="elu", strides=(1, 1), padding='VALID', name=name)
            self.sub_layers[name] = layer

        x = layer(inputs, training=training)
        x = self.batch_normalization(x, training=training, name=name+"-BN")
        return x

    def depthwise_conv2d(self, inputs, deptch_multiplier, name, training, kernel_size, activation="elu"):
        layer = self.sub_layers.get(name)
        if not layer:
            layer = tf.keras.layers.DepthwiseConv2D(kernel_size, padding='VALID', depth_multiplier=deptch_multiplier, activation=activation, name=name)
            self.sub_layers[name] = layer

        x = layer(inputs, training=training)
        x = self.batch_normalization(x, training=training, name=name + "-BN")
        return x

    def separable_conv2d(self, inputs, channel, name, training, kernel_size, depth_multiplier):
        layer = self.sub_layers.get(name)
        if not layer:
            layer = tf.keras.layers.SeparableConv2D(channel, kernel_size, use_bias=False, depth_multiplier=depth_multiplier,
                                                    activation="elu", strides=(1, 1), padding='VALID', name=name)
            self.sub_layers[name] = layer

        x = layer(inputs, training=training)
        x = self.batch_normalization(x, training=training, name=name + "-BN")
        return x

    def call(self, inputs, training, **kwargs):

        is_training = training
        K = self.k
        D = self.d
        C = self.c
        shape = tf.shape(inputs)
        N = shape[0]
        P = shape[1] if self.p < 0 else self.p
        tag = "XConvLayerCoreV1-"
        depth_multiplier = self.depth_multiplier
        with_global = self.with_global

        pts = inputs[:, :, :3]  # (B, N, 3)
        if self.p < 0:
            qrs = pts
        else:
            qrs = pts[:, :self.p, :]

        _, indices_dilated = pf.knn_indices_general(qrs, pts, K * D, True)
        indices = indices_dilated[:, :, ::self.d, :]

        # TODO: Sorting

        nn_pts = tf.gather_nd(pts, indices, name=tag + 'nn_pts')  # (N, P, K, 3)
        nn_pts_center = tf.expand_dims(qrs, axis=2, name=tag + 'nn_pts_center')  # (N, P, 1, 3)
        nn_pts_local = tf.subtract(nn_pts, nn_pts_center, name=tag + 'nn_pts_local')  # (N, P, K, 3)

        # Prepare features to be transformed
        nn_fts_from_pts_0 = self.dense(nn_pts_local, self.cpf, tag + 'nn_fts_from_pts_0', is_training)
        nn_fts_from_pts = self.dense(nn_fts_from_pts_0, self.cpf, tag + 'nn_fts_from_pts', is_training)

        nn_fts_from_prev = tf.gather_nd(inputs, indices, name=tag + 'nn_fts_from_prev')[:, :, :, 3:]
        nn_fts_input = tf.concat([nn_fts_from_pts, nn_fts_from_prev], axis=-1, name=tag + 'nn_fts_input')

        X_0 = self.conv2d(nn_pts_local, self.k * self.k, tag + 'X_0', is_training, (1, K))
        X_0_KK = tf.reshape(X_0, (N, P, K, K), name=tag + 'X_0_KK')
        X_1 = self.depthwise_conv2d(X_0_KK, K, tag + 'X_1', is_training, (1, K))
        X_1_KK = tf.reshape(X_1, (N, P, K, K), name=tag + 'X_1_KK')
        X_2 = self.depthwise_conv2d(X_1_KK, K, tag + 'X_2', is_training, (1, K), activation=None)
        X_2_KK = tf.reshape(X_2, (N, P, K, K), name=tag + 'X_2_KK')
        fts_X = tf.matmul(X_2_KK, nn_fts_input, name=tag + 'fts_X')

        fts_conv = self.separable_conv2d(fts_X, C, tag + 'fts_conv', is_training, (1, K),
                                       depth_multiplier=depth_multiplier)
        fts_conv_3d = tf.squeeze(fts_conv, axis=2, name=tag + 'fts_conv_3d')

        if with_global:
            fts_global_0 = self.dense(qrs, C // 4, tag + 'fts_global_0', is_training)
            fts_global = self.dense(fts_global_0, C // 4, tag + 'fts_global', is_training)
            fts_conv_3d = tf.concat([fts_global, fts_conv_3d], axis=-1, name=tag + 'fts_conv_3d_with_global')
        else:
            pass

        return tf.concat([qrs, fts_conv_3d], axis=-1, name=tag+"xconv_output")



class XConvLayerCoreV2(tf.keras.layers.Layer):
    """
    The X-Conv kernel used in "PointCNN"(https://arxiv.org/abs/1801.07791)
    """
    def __init__(self, p, k, d, c, cpf, depth_multiplier=2, sampling="random", sorting_method=None, with_global=False, **kwargs):
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
        if with_global:
            self.l_global_pos2feature = (
                tf.keras.layers.Dense(self.c // 4, activation="elu"),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dense(self.c // 4, activation="elu"),
                tf.keras.layers.BatchNormalization()
            )
        else:
            self.l_global_pos2feature = None

    def call(self, inputs, *args, **kwargs):
        c = ComputationContext(*args, **kwargs)
        p = inputs[:, :, :3]  # (B, N, 3)
        f = inputs[:, :, 3:]  # (B, N, F)

        shape = tf.shape(inputs)
        B = shape[0]  # Batch size
        N = self.p if self.p > 0 else shape[1]  # Number of points for output

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

        # Concat the origin feature, cannot concat when it only has position feature
        f_ = tf.concat([f_, tf.gather_nd(inputs, indices)[:, :, :, 3:]], axis=-1)  # (B, p, k, cpf) ~concat~ (B, p, k, F) = (B, p, k, cpf + F)

        # X convolution core
        x_conv_mat = tf.reshape(c(self.l_xconv1, p_), (B, N, self.k, self.k))  # (B, p, k, 3) -> (B, p, k, k)
        x_conv_mat = tf.reshape(c(self.l_xconv2, x_conv_mat), (B, N, self.k, self.k))  # (B, p, k, k) -> (B, p, k, k)
        x_conv_mat = tf.reshape(c(self.l_xconv3, x_conv_mat), (B, N, self.k, self.k))  # (B, p, k, k) -> (B, p, k, k)

        # Matrix multiplication
        f_ = tf.linalg.matmul(x_conv_mat, f_)  # (B, p, k, k) x (B, p, k, cpf + f) = (B, p, k, cpf + f)

        # Final convolution
        f_ = tf.squeeze(c(self.l_final_conv, f_), axis=2)  # (B, p, k, cpf + f) -> (B, p, 1, c) ->(squeeze)-> (B, p, c)

        # With global feature
        if self.with_global:
            f_global = c(self.l_global_pos2feature, q)  # (B, p, 3) -> (B, p, c/4)
            f_ = tf.concat([f_global, f_], axis=-1)  # (B, p, c/4) ~concat~ (B, p, c) = (B, p, c + c/4)

        return tf.concat([q, f_], axis=-1)  # (B, p, c + c/4) ~concat~ (B, p, 3) = (B, p, 3 + c + c/4)


XConvPoolingLayer = XConvLayerCoreV1


class XConvLayer(XConvLayerCoreV1):
    """
    The actual convolution layer. Different from the XConvLayerCore, it doesn't take "p" as a parameter, it maps
    an input of (BxNxF) tensor to (BxNxc) tensor.
    """
    def __init__(self, *args, **kwargs):
        super(XConvLayer, self).__init__(-1, *args, **kwargs)


class FeatureReshapeLayer(tf.keras.layers.Layer):
    """
    A feature resize layer that reshape the feature dimension using several dense layer with dropout
    """
    def __init__(self, channels, dropout, **kwargs):
        super(FeatureReshapeLayer, self).__init__()

        self.channels = channels
        self.dropout = dropout
        self.layers = []

        for channel_size, dropout_rate in zip(channels, dropout):
            self.layers.append(tf.keras.layers.Dense(channel_size))
            if dropout_rate > 0.0:
                self.layers.append(tf.keras.layers.Dropout(dropout_rate))
            self.layers.append(tf.keras.layers.BatchNormalization())

    def call(self, inputs, *args, **kwargs):
        c = ComputationContext(*args, **kwargs)
        p = inputs[:, :, :3]  # (B, N, 3)
        f = inputs[:, :, 3:]  # (B, N, F)

        f_ = c(self.layers, f)  # (B, N, F) -> (B, N, self.channels[-1])

        return tf.concat([p, f_], axis=-1)


def layer_from_config(layer_conf):
    """
    Create a layer from configuration
    :param conf: The configuration dict, where it should have an "name" entry specified the name of
    the layer and other parameter to initialize the layer
    :return: A corresponding keras layer
    """
    layer_map = {
        "conv-xconv": XConvLayer,
        "pooling-xconv": XConvPoolingLayer,
        "feature-reshape": FeatureReshapeLayer
    }

    assert layer_conf["name"] in layer_map, "Did not find layer with name \"{}\"".format(layer_conf["name"])
    return layer_map[layer_conf["name"]](**layer_conf)