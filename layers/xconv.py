import tensorflow as tf
import legacy.pointfly as pf
from utils.confutil import register_conf


class XConvLayerCoreV1(tf.keras.layers.Layer):
    """
    The X-Conv kernel used in "PointCNN"(https://arxiv.org/abs/1801.07791)
    """
    def __init__(self, p, k, d, c, cpf, depth_multiplier=2, sampling="random",
                 sorting_method=None, with_global=False, label=None, **kwargs):
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
        :param label: An optional label for the layer
        """
        super(XConvLayerCoreV1, self).__init__(name=label)
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

    def batch_normalization(self, inputs, name, training):
        layer = self.sub_layers.get(name)
        if not layer:
            layer = tf.keras.layers.BatchNormalization(momentum=0.99, name=name)
            self.sub_layers[name] = layer
        return layer(inputs, training=training)

    def dense(self, inputs, units, name, training, activation="elu"):
        layer = self.sub_layers.get(name)
        if not layer:
            layer = tf.keras.layers.Dense(units, use_bias=False, activation=activation, name=name)
            self.sub_layers[name] = layer

        x = layer(inputs, training=training)
        x = self.batch_normalization(x, training=training, name=name+"-BN")
        return x

    def conv2d(self, inputs, channel, name, training, kernel_size, activation="elu"):
        layer = self.sub_layers.get(name)
        if not layer:
            layer = tf.keras.layers.Conv2D(channel, kernel_size, use_bias=False, activation=activation, strides=(1, 1), padding='VALID', name=name)
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

    def separable_conv2d(self, inputs, channel, name, training, kernel_size, depth_multiplier, activation="elu"):
        layer = self.sub_layers.get(name)
        if not layer:
            layer = tf.keras.layers.SeparableConv2D(channel, kernel_size, use_bias=False, depth_multiplier=depth_multiplier,
                                                    activation=activation, strides=(1, 1), padding='VALID', name=name)
            self.sub_layers[name] = layer

        x = layer(inputs, training=training)
        x = self.batch_normalization(x, training=training, name=name + "-BN")
        return x

    def count_params(self):
        return sum([layer.count_params() for layer in self.sub_layers.values()])

    def call(self, inputs, training, **kwargs):
        pts = inputs[0]  # pts: (B, N, 3)
        features = inputs[1] # fts: (B, N, F)

        is_training = training
        K = self.k
        D = self.d
        C = self.c
        shape = tf.shape(pts)
        N = shape[0]
        P = shape[1] if self.p < 0 else self.p
        tag = "XConvLayerCoreV1-"
        depth_multiplier = self.depth_multiplier
        with_global = self.with_global

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

        # Use concat the gather_nd to break down when F = 0 (no any features)
        nn_fts_from_prev = tf.gather_nd(tf.concat([pts, features], axis=-1), indices, name=tag + 'nn_fts_from_prev')[:, :, :, 3:]
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

        return qrs, fts_conv_3d


@register_conf(name="pooling-xconv", scope="layer", conf_func="self")
class XConvPoolingLayer(XConvLayerCoreV1):
    pass


@register_conf(name="conv-xconv", scope="layer", conf_func="self")
class XConvLayer(XConvLayerCoreV1):
    """
    The actual convolution layer. Different from the XConvLayerCore, it doesn't take "p" as a parameter, it maps
    an input of (BxNxF) tensor to (BxNxc) tensor.
    """
    def __init__(self, *args, **kwargs):
        super(XConvLayer, self).__init__(-1, *args, **kwargs)