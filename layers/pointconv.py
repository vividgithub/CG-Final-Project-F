import tensorflow as tf
from utils.confutil import register_conf
from tf_ops.sampling.tf_sampling import farthest_point_sample


kernel_regularizer = tf.keras.regularizers.L1L2(l2=1e-4)
bias_regularizer = tf.keras.regularizers.L1L2(l2=1e-4)
kernel_initializer = tf.keras.initializers.he_uniform()
bias_initializer = tf.keras.initializers.he_uniform()


@register_conf(name="print", scope="layer", conf_func="self")
class PrintLayer(tf.keras.layers.Layer):
    def __init__(self, points=False, features=True, **kwargs):
        super(PrintLayer, self).__init__()
        self.points = points
        self.features = features

    def call(self, inputs, *args, **kwargs):
        import sys, logger
        outputs = [self.points, self.features]
        for name, input, output in zip(("points", "features"), inputs, outputs):
            if output:
                logger.log(name, color="red")
                logger.log(input, color="red")
                tf.print(input, output_stream=sys.stdout)
        return inputs


@register_conf(name="point-reshape", scope="layer", conf_func="self")
class PointReshapeLayer(tf.keras.layers.Layer):
    def __init__(self, shape, **kwargs):
        super(PointReshapeLayer, self).__init__()
        self.shape = shape

    def call(self, inputs, training, **kwargs):
        features = inputs[1]
        layer = tf.keras.layers.Reshape((self.shape,))
        output = layer(features)
        return inputs[0], output


@register_conf(name="point-deconv", scope="layer", conf_func="self")
class PointDeconvLayer(tf.keras.layers.Layer):
    def __init__(self, out_channels, dropout_rate=0.4, **kwargs):
        super(PointDeconvLayer, self).__init__()
        self.out_channels = out_channels
        self.dropout_rate = dropout_rate
        self.sub_layers = dict()

    def batch_normalization(self, inputs, name, training):
        layer = self.sub_layers.get(name)
        if not layer:
            layer = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.1, name=name)
            self.sub_layers[name] = layer
        return layer(inputs, training=training)

    def relu(self, inputs, name, training):
        layer = self.sub_layers.get(name)
        if not layer:
            layer = tf.keras.layers.ReLU(name=name)
            self.sub_layers[name] = layer
        return layer(inputs)

    def dropout(self, inputs, p, name, training):
        layer = self.sub_layers.get(name)
        if not layer:
            layer = tf.keras.layers.Dropout(p, name=name)
            self.sub_layers[name] = layer
        return layer(inputs, training=training)

    def dense(self, inputs, units, name, training):
        layer = self.sub_layers.get(name)
        if not layer:
            layer = tf.keras.layers.Dense(units, activation="linear", name=name,
                    kernel_regularizer=kernel_regularizer,
                    bias_regularizer=bias_regularizer,
                    kernel_initializer=kernel_initializer,
                    bias_initializer=bias_initializer)
            self.sub_layers[name] = layer

        x = layer(inputs, training=training)
        x = self.batch_normalization(x, training=training, name=name+"-BN")
        x = self.relu(x, training=training, name=name+"-relu")
        x = self.dropout(x, self.dropout_rate, name=name+"-dropout", training=training)
        return x

    def call(self, inputs, training, **kwargs):
        points = inputs[1]
        is_training = training
        tag = "PointDeconvLayer-"

        output = self.dense(points, self.out_channels, tag + 'dense', is_training)

        return inputs[0], output


@register_conf(name="point-conv", scope="layer", conf_func="self")
class PointConvLayer(tf.keras.layers.Layer):
    def __init__(self, npoint, nsample, in_channel, mlp, bandwidth, group_all, use_features=True, **kwargs):
        super(PointConvLayer, self).__init__()
        self.npoint = npoint
        self.nsample = nsample
        self.in_channel = in_channel
        self.mlp = mlp
        self.bandwidth = bandwidth
        self.group_all = group_all
        self.use_features = use_features
        self.sub_layers = dict()

    def batch_normalization(self, inputs, name, training):
        layer = self.sub_layers.get(name)
        if not layer:
            layer = tf.keras.layers.BatchNormalization(epsilon=1e-5, momentum=0.1, name=name, axis=1)
            self.sub_layers[name] = layer
        return layer(inputs, training=training)

    def conv1d(self, inputs, filters, kernel_size, name, training):
        layer = self.sub_layers.get(name)
        if not layer:
            layer = tf.keras.layers.Conv1D(filters, kernel_size, name=name, data_format='channels_first',
                    kernel_regularizer=kernel_regularizer,
                    bias_regularizer=bias_regularizer,
                    kernel_initializer=kernel_initializer,
                    bias_initializer=bias_initializer)
            self.sub_layers[name] = layer

        x = layer(inputs, training=training)
        x = self.batch_normalization(x, training=training, name=name+"-BN")
        return x

    def conv2d(self, inputs, filters, kernel_size, name, training):
        layer = self.sub_layers.get(name)
        if not layer:
            layer = tf.keras.layers.Conv2D(filters, kernel_size, name=name, data_format='channels_first',
                    kernel_regularizer=kernel_regularizer,
                    bias_regularizer=bias_regularizer,
                    kernel_initializer=kernel_initializer,
                    bias_initializer=bias_initializer)
            self.sub_layers[name] = layer

        x = layer(inputs, training=training)
        x = self.batch_normalization(x, training=training, name=name+"-BN")
        return x

    def relu(self, inputs, name, training):
        layer = self.sub_layers.get(name)
        if not layer:
            layer = tf.keras.layers.ReLU(name=name)
            self.sub_layers[name] = layer
        return layer(inputs)

    def density_net(self, inputs, hidden_unit, name, training):
        hidden_unit = hidden_unit + [1]
        density_scale = tf.expand_dims(inputs, 1)  # [B, 1, N]
        for i in range(len(hidden_unit)):
            density_scale = self.conv1d(density_scale, hidden_unit[i], 1, name+"conv1d-"+str(i), training)
            density_scale = self.relu(density_scale, name+"relu-"+str(i), training)
        return density_scale  # [B, 1, N]

    def weight_net(self, inputs, out_channel, hidden_unit, name, training):
        hidden_unit = [out_channel] if hidden_unit is None else hidden_unit + [out_channel]
        weights = inputs  # [B, C, N, 1]
        for i in range(len(hidden_unit)):
            weights = self.conv2d(weights, hidden_unit[i], 1, name+"conv2d-"+str(i), training)
            weights = self.relu(weights, name+"relu-"+str(i), training)
        return weights  # [B, out_channel, N, 1]

    @staticmethod
    def square_distance(src, dst):
        """
        Calculate Euclid distance between each two points.
        src^T * dst = xn * xm + yn * ym + zn * zm；
        sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
        sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
        dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
             = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
        Input:
            src: source points, [B, N, C]
            dst: target points, [B, M, C]
        Output:
            dist: per-point square distance, [B, N, M]
        """
        B, N, _ = src.shape
        _, M, _ = dst.shape
        dist = -2 * tf.matmul(src, tf.transpose(dst, (0, 2, 1)))
        dist += tf.expand_dims(tf.reduce_sum(tf.square(src), -1), 2)
        dist += tf.expand_dims(tf.reduce_sum(tf.square(dst), -1), 1)
        return dist

    @staticmethod
    def index_points(points, idx):
        """
        Input:
            points: input points data, [B, N, C]
            idx: sample index data, [B, S]
        Return:
             new_points: indexed points data, [B, S, C]
        """
        new_points = tf.gather(points, idx, axis=1, batch_dims=1)
        return new_points

    @staticmethod
    def farthest_point_sample(xyz, npoint):
        """
        Input:
            xyz: point cloud data, [B, N, C]
            npoint: number of samples
        Return:
             centroids: sampled point cloud index, [B, npoint]
        """
        return None

    @staticmethod
    def compute_density(xyz, bandwidth):
        """xyz: input points position data, [B, N, C]"""
        sqrdists = PointConvLayer.square_distance(xyz, xyz)  # [B, N, N]
        gaussian_density = tf.exp(- sqrdists / (2.0 * bandwidth * bandwidth)) / (2.5 * bandwidth)
        xyz_density = tf.reduce_mean(gaussian_density, -1)  # [B, N]

        return xyz_density

    @staticmethod
    def knn_point(nsample, xyz, new_xyz):
        """
        Input:
            nsample: max sample number in local region
            xyz: all points, [B, N, C]
            new_xyz: query points, [B, S, C]
        Return:
            grouped_idx: grouped points index, [B, S, nsample]
        """
        sqrdists = PointConvLayer.square_distance(new_xyz, xyz)
        _, grouped_idx = tf.nn.top_k(-sqrdists, nsample, sorted=False)
        # grouped_idx = tf.sort(grouped_idx)  # XXX: sort for check
        return grouped_idx

    @staticmethod
    def sample_and_group(npoint, nsample, xyz, points, density_scale=None):
        """
        Input:
            xyz: input points position data, [B, N, C]
            points: input points data, [B, N, D]
        Return:
            new_xyz: sampled points position data, [B, npoint, C]
            new_points: sampled points data, [B, npoint, nsample, C+D]
        """
        fps_idx = farthest_point_sample(npoint, xyz)
        new_xyz = PointConvLayer.index_points(xyz, fps_idx)  # [B, npoint, C]
        idx = PointConvLayer.knn_point(nsample, xyz, new_xyz)
        grouped_xyz = PointConvLayer.index_points(xyz, idx)  # [B, npoint, nsample, C]
        grouped_xyz_norm = grouped_xyz - tf.expand_dims(new_xyz, 2)
        if points is not None:
            grouped_points = PointConvLayer.index_points(points, idx)
            new_points = tf.concat([grouped_xyz_norm, grouped_points], -1)  # [B, npoint, nsample, C+D]
        else:
            new_points = grouped_xyz_norm

        if density_scale is None:
            return new_xyz, new_points, grouped_xyz_norm, idx
        else:
            grouped_density = PointConvLayer.index_points(density_scale, idx)
            return new_xyz, new_points, grouped_xyz_norm, idx, grouped_density

    @staticmethod
    def sample_and_group_all(xyz, points, density_scale=None):
        """
        Input:
            xyz: input points position data, [B, N, C]
            points: input points data, [B, N, D]
            density_scale: density scale, [B, N, 1]
        Return:
            new_xyz: sampled points position data, [B, 1, C]
            new_points: sampled points data, [B, 1, N, C+D]
            grouped_xyz: [B, 1, N, C]
            grouped_density: [B, 1, N, 1]
        """
        new_xyz = tf.reduce_mean(xyz, 1)  # [B, C]
        grouped_xyz = tf.expand_dims(xyz, 1) - tf.expand_dims(tf.expand_dims(new_xyz, 1), 1)
        if points is not None:
            new_points = tf.concat([grouped_xyz, tf.expand_dims(points, 1)], -1)
        else:
            new_points = grouped_xyz
        if density_scale is None:
            return new_xyz, new_points, grouped_xyz
        else:
            grouped_density = tf.expand_dims(density_scale, 1)
            return new_xyz, new_points, grouped_xyz, grouped_density

    def reshape(self, inputs, shape, name, training):
        layer = self.sub_layers.get(name)
        if not layer:
            layer = tf.keras.layers.Reshape(shape)
            self.sub_layers[name] = layer
        return layer(inputs)

    def dense(self, inputs, units, name, training, activation="relu"):
        layer = self.sub_layers.get(name)
        if not layer:
            layer = tf.keras.layers.Dense(units, activation=activation, name=name,
                    kernel_regularizer=kernel_regularizer,
                    bias_regularizer=bias_regularizer,
                    kernel_initializer=kernel_initializer,
                    bias_initializer=bias_initializer)
            self.sub_layers[name] = layer

        x = layer(inputs, training=training)
        x = self.batch_normalization(x, training=training, name=name+"-BN")
        return x

    def call(self, inputs, training, **kwargs):
        xyz = inputs[0]  # [B, N, C=3]
        points = inputs[1]  # [B, N, D]
        if not self.use_features:
            points = None

        is_training = training
        tag = "PointConvLayer-"

        xyz_density = PointConvLayer.compute_density(xyz, self.bandwidth)  # [B, N]
        density_scale = self.density_net(xyz_density, [8, 8], tag + "densitynet", is_training)  # [B, 1, N]

        if self.group_all:
            new_xyz, new_points, grouped_xyz_norm, grouped_density = \
                PointConvLayer.sample_and_group_all(xyz, points, tf.transpose(density_scale, (0, 2, 1)))
        else:
            new_xyz, new_points, grouped_xyz_norm, _, grouped_density = \
                PointConvLayer.sample_and_group(self.npoint, self.nsample, xyz, points, tf.transpose(density_scale, (0, 2, 1)))

        new_points = tf.transpose(new_points, (0, 3, 2, 1))  # [B, C+D, nsample, npoint]
        for i in range(len(self.mlp)):
            new_points = self.conv2d(new_points, self.mlp[i], 1, tag + "conv2d-"+str(i), is_training)
            new_points = self.relu(new_points, tag + "relu-"+str(i), is_training)  # [B, Cout, nsample, npoint]

        grouped_xyz = tf.transpose(grouped_xyz_norm, (0, 3, 2, 1))  # [B, C, nsample, npoint]
        grouped_xyz = tf.multiply(grouped_xyz, tf.transpose(grouped_density, (0, 3, 2, 1)))
        weights = self.weight_net(grouped_xyz, 16, [8, 8], tag + "weightnet", is_training)  # [B, 16, nsample, npoint]
        new_points = tf.matmul(tf.transpose(new_points, (0, 3, 1, 2)), tf.transpose(weights, (0, 3, 2, 1)))  # [B, npoint, Cout, 16]
        new_points = self.reshape(new_points, (self.npoint, self.mlp[-1] * 16), tag + "reshape", is_training)  # [B, npoint, 16*Cout]
        new_points = self.dense(new_points, self.mlp[-1], tag + "linear", is_training)  # [B, npoint, Cout]
        new_points = self.relu(new_points, tag + "linear-relu", is_training)

        return new_xyz, new_points


@register_conf(name="pointconv-output", scope="layer", conf_func="self")
class PointconvOutputLayer(tf.keras.layers.Layer):
    def __init__(self, class_count, use_position=False, label=None, **kwargs):
        super(PointconvOutputLayer, self).__init__(name=label)
        self.use_position = use_position
        self.dense = tf.keras.layers.Dense(
            class_count,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer)
        self.output_size = kwargs['output_size']

    def call(self, inputs, training, *args, **kwargs):
        x = tf.concat(inputs, axis=-1) if self.use_position else inputs[1]  # (B, N, F) or (B, N, F + 3)
        x = tf.tile(tf.expand_dims(x, 1), [1, self.output_size, 1])
        if not training:
            x = tf.reduce_mean(x, axis=1, keepdims=True)  # (B, N, F) for training and (B, 1, F) for testing
        x = self.dense(x, *args, **kwargs)  # (B, N/1, class_count)
        return x

    def count_params(self):
        return self.dense.count_params()

