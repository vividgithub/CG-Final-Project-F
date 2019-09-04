import tensorflow as tf
import legacy.pointfly as pf

class XConvLayer(tf.keras.layers.Layer):

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
        super(XConvLayer, self).__init__()
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

        # l_dense1 and l_dense2 used in converting the position features
        self.l_dense1 = tf.keras.layers.Dense(self.cpf)
        self.l_bn1 = tf.keras.layers.BatchNormalization()
        self.l_dense2 = tf.keras.layers.Dense(self.cpf)
        self.l_bn2 = tf.keras.layers.BatchNormalization()

        # The X-conv core, it uses 3 convolution kernel to transform the local coordinate (B, N, k, 3)
        # ->(conv1, bn) -> (B, N, k, k) ->(conv2, bn) -> (B, N, k, k) ->(conv3, bn) -> (B, N, k, k)
        self.l_conv1 = tf.keras.layers.Conv2D(self.k * self.k, (1, self.k), activation="elu")
        self.l_bn3 = tf.keras.layers.BatchNormalization()
        self.l_conv2 = tf.keras.layers.DepthwiseConv2D((1, self.k), depth_multiplier=self.k, activation="elu")
        self.l_bn4 = tf.keras.layers.BatchNormalization()
        self.l_conv3 = tf.keras.layers.DepthwiseConv2D((1, self.k), depth_multiplier=self.k, activation="elu")
        self.l_bn5 = tf.keras.layers.BatchNormalization()

        # Final convolution
        self.l_conv4 = tf.keras.layers.SeparableConv2D(self.c, (1, self.k), depth_multiplier=self.depth_multiplier, activation="elu")
        self.l_bn6 = tf.keras.layers.BatchNormalization()

    def call(self, inputs, **kwargs):
        pts = inputs[:, :, :3]  # (B, N, 3)
        fts = inputs[:, :, 3:]  # (B, N, F)

        shape = tf.shape(inputs)
        B = shape[0]
        N = shape[1]

        # Get the sampling points
        # Not so randomly, only follow the original implementations
        qrs = pts if self.p < 0 else pts[:, :self.p, :]  # (B, p, 3)

        # Get the neighborhood indices
        _, indices_dilated = pf.knn_indices_general(qrs, pts, self.k * self.d, True)  # (B, p, k*d, 2)
        indices = indices_dilated[:, :, ::self.d, :]  # (B, p, k, 2)

        # Sort the points
        # TODO:

        # Group and convert to local coordinate
        nn_pts = tf.gather_nd(pts, indices)  # (B, p, k, 3)
        nn_pts_center = tf.expand_dims(qrs, axis=2)  # (B, p, 1, 3)
        nn_pts_local = tf.subtract(nn_pts, nn_pts_center)  # (B, p, k, 3)

        # Convert position into features
        nn_fts_from_pts_0 = self.l_dense1(nn_pts_local, **kwargs)  # (B, p, k, cpf)
        nn_fts_from_pts_0 = self.l_bn1(nn_fts_from_pts_0, **kwargs)  # (B, p, k, cpf)

        nn_fts_from_pts = self.l_dense2(nn_fts_from_pts_0, **kwargs)  # (B, p, k, cpf)
        nn_fts_from_pts = self.l_bn2(nn_fts_from_pts, **kwargs)  # (B, p, k, cpf)

        nn_fts_from_prev = tf.gather_nd(fts, indices)  # (B, p, k, f)
        nn_fts_input = tf.concat([nn_fts_from_pts, nn_fts_from_prev], axis=-1)  # (B, p, k, cpf + f)

        # X convolution core
        X_0 = self.l_conv1(nn_pts_local, **kwargs)
        X_0 = self.l_bn3(X_0, **kwargs)
        X_0_KK = tf.reshape(X_0, (B, N, self.k, self.k))

        X_1 = self.l_conv2(X_0_KK, **kwargs)
        X_1 = self.l_bn4(X_1, **kwargs)
        X_1_KK = tf.reshape(X_1, (B, N, self.k, self.k))

        X_2 = self.l_conv3(X_1_KK, **kwargs)
        X_2 = self.l_bn5(X_2, **kwargs)
        X_2_KK = tf.reshape(X_2, (B, N, self.k, self.k))

        fts_X = tf.linalg.matmul(X_2_KK, nn_fts_input)  # (B, p, k, cpf + f)

        # Final convolution
        fts_conv = self.l_conv4(fts_X, **kwargs)  # (B, p, 1, c)
        fts_conv = self.l_bn6(fts_conv, **kwargs)  # (B, p, 1, c)
        fts_conv_3d = tf.squeeze(fts_conv, axis=2)  # (B, p, c)

        # With global feature
        # TODO:

        return fts_conv_3d