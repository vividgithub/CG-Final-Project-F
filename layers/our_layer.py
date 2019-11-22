import tensorflow as tf
from tf_sampling import farthest_point_sample, gather_point
from tf_grouping import query_ball_point, group_point, knn_point
from tf_interpolate import three_nn, three_interpolate

class SALayerCore(tf.keras.layers.Layer):
    """
    The X-Conv kernel used in "PointCNN"(https://arxiv.org/abs/1801.07791)
    """
    def __init__(self, npoint, radius, nsample, mlp, is_training, bn=True, pooling='max', knn=False, use_xyz=True, use_nchw=False, label=None, **kwargs):
        super(SALayerCore, self).__init__(name=label)
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp = mlp
        self.is_training = is_training
        self.bn = bn
        self.pooling = pooling
        self.knn = knn
        self.use_xyz = use_xyz
        self.use_nchw = use_nchw

    def sample_and_group(self, xyz, points): 
        '''
        Input:
            npoint: int32
            radius: float32
            nsample: int32
            xyz: (batch_size, ndataset, 3) TF tensor
            points: (batch_size, ndataset, channel) TF tensor, if None will just use xyz as points
            knn: bool, if True use kNN instead of radius search
            use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
        Output:
            new_xyz: (batch_size, npoint, 3) TF tensor
            new_points: (batch_size, npoint, nsample, 3+channel) TF tensor
            idx: (batch_size, npoint, nsample) TF tensor, indices of local points as in ndataset points
            grouped_xyz: (batch_size, npoint, nsample, 3) TF tensor, normalized point XYZs
                (subtracted by seed point XYZ) in local regions
        '''
        # 每个batch取样的最远的1024个点
        # new_xyz: b * npoints * 3
        new_xyz = gather_point(xyz, farthest_point_sample(self.npoint, xyz))  # (batch_size, npoint, 3)
        if self.knn:
            _,idx = knn_point(self.nsample, xyz, new_xyz)
        else:
            # idx: (batch_size, npoint, nsample) int32 array, indices to input points
            # pts_cnt: (batch_size, npoint) int32 array, number of unique points in each local region
            idx, pts_cnt = query_ball_point(self.radius, self.nsample, xyz, new_xyz)
        # grouped_xyz: (batch_size, npoint, nsample, channel)
        # according to idx return corresponding chanel
        grouped_xyz = group_point(xyz, idx)  # (batch_size, npoint, nsample, 3)
        # move the points to the center (by minusing the coordinate of the center)
        grouped_xyz -= tf.tile(tf.expand_dims(new_xyz, 2), [1,1,self.nsample,1])  # translation normalization
        if points is not None:
            grouped_points = group_point(points, idx)  # (batch_size, npoint, nsample, channel)
            if self.use_xyz:
                new_points = tf.concat([grouped_xyz, grouped_points], axis=-1)  # (batch_size, npoint, nample, 3+channel)
            else:
                new_points = grouped_points
        else:
            new_points = grouped_xyz

        return new_xyz, new_points, idx, grouped_xyz

    def conv2d(self, 
               inputs,
               num_output_channels,
               kernel_size,
               num,
               stride=[1, 1],
               padding='SAME',
               data_format='NHWC',
               use_xavier=True,
               stddev=1e-3,
               weight_decay=0.0,
               activation_fn="relu",
               is_dist=False,
               initial_bias=0.0):
        """ 2D convolution with non-linear operation.

        Args:
            inputs: 4-D tensor variable BxHxWxC
            num_output_channels: int
            kernel_size: a list of 2 ints
            scope: string
            stride: a list of 2 ints
            padding: 'SAME' or 'VALID'
            use_xavier: bool, use xavier_initializer if true
            stddev: float, stddev for truncated_normal init
            weight_decay: float
            activation_fn: function
            bn: bool, whether to use batch norm
            bn_decay: float or float tensor variable in [0,1]
            is_training: bool Tensor variable

        Returns:
            Variable tensor
        """
        # 定义命名空间
        kernel_h, kernel_w = kernel_size
        num_in_channels = inputs.get_shape()[-1].value
        kernel_shape = [kernel_h, kernel_w,
                        num_in_channels, num_output_channels]
        # kernel: 卷积核
        # 要求是一个Tensor，
        # 具有[filter_height, filter_width, in_channels, out_channels]这样的shape，
        # 具体含义是[卷积核的高度，卷积核的宽度，图像通道数，卷积核个数]，
        # 要求类型与参数input相同，有一个地方需要注意，第三维in_channels，
        # 就是参数input的第四维

        # 返回使用xavier initializer生成的对应shape的variable
        if use_xavier:
            kernel_initializer = tf.keras.initializers.glorot_uniform()
        else:
            kernel_initializer = tf.keras.initializers.TruncatedNormal(stddev=stddev)

        bias_initializer = tf.keras.initializers.Constant(value=initial_bias)

        kernel_regularizer = tf.keras.regularizers.l2(0.)

        stride_h, stride_w = stride

        layer = tf.keras.layers.Conv2D(num_output_channels, 
                                       kernel_size, 
                                       use_bias=True, 
                                       strides=(1, 1), 
                                       padding='valid',
                                       data_format=data_format,
                                       name='conv%d'%(num),
                                       kernel_initializer=kernel_initializer,
                                       bias_initializer=bias_initializer,
                                       kernel_regularizer=kernel_regularizer)

        x = layer(inputs, training=self.is_training)
        if self.bn:
            assert data_format == 'channels_last'
            x = self.batch_normalization(x, training=self.is_training, name="BN%d"%(num))

        if self.activation_fn == "relu":
            layer = tf.keras.layers.ReLU(max_value=None, negative_slope=0.0, threshold=0.0)
            x = layer(x, training=self.is_training)

        return x

    def batch_normalization(self, inputs, name, training):
        layer = tf.keras.layers.BatchNormalization(momentum=0.9, name=name)
        return layer(inputs, training=training)

    def call(self, xyz, points, name, **kwargs):
        data_format = 'channels_first' if self.use_nchw else 'channels_last'

        new_xyz, new_points, idx, grouped_xyz = self.sample_and_group(xyz, points)

        if self.use_nchw: 
            new_points = tf.transpose(new_points, [0,3,1,2])

        for i, num_out_channel in enumerate(self.mlp):
            new_points = self.conv2d(new_points, 
                                     num_out_channel, 
                                     [1,1],
                                     i,
                                     padding='VALID', 
                                     stride=[1,1],
                                     data_format=data_format)

        if self.use_nchw: 
            new_points = tf.transpose(new_points, [0,2,3,1]) 

        # Pooling in Local Regions
        # pooling = max
        if self.pooling=='max':
            new_points = tf.reduce_max(new_points, axis=[2], keep_dims=True, name='maxpool')

        new_points = tf.squeeze(new_points, [2])  # (batch_size, npoints, mlp2[-1])
        return new_xyz, new_points, idx


class FPLayerCore(tf.keras.layers.Layer):
    """
    The X-Conv kernel used in "PointCNN"(https://arxiv.org/abs/1801.07791)
    """
    def __init__(self, mlp, is_training, bn=True, label=None, **kwargs):
        super(SALayerCore, self).__init__(name=label)
        self.mlp = mlp
        self.is_training = is_training
        self.bn = bn

    def conv2d(self, 
               inputs,
               num_output_channels,
               kernel_size,
               num,
               stride=[1, 1],
               padding='SAME',
               data_format='NHWC',
               use_xavier=True,
               stddev=1e-3,
               weight_decay=0.0,
               activation_fn="relu",
               is_dist=False,
               initial_bias=0.0,
               ):
        """ 2D convolution with non-linear operation.

        Args:
            inputs: 4-D tensor variable BxHxWxC
            num_output_channels: int
            kernel_size: a list of 2 ints
            scope: string
            stride: a list of 2 ints
            padding: 'SAME' or 'VALID'
            use_xavier: bool, use xavier_initializer if true
            stddev: float, stddev for truncated_normal init
            weight_decay: float
            activation_fn: function
            bn: bool, whether to use batch norm
            bn_decay: float or float tensor variable in [0,1]
            is_training: bool Tensor variable

        Returns:
            Variable tensor
        """
        # 定义命名空间
        kernel_h, kernel_w = kernel_size
        num_in_channels = inputs.get_shape()[-1].value
        kernel_shape = [kernel_h, kernel_w,
                        num_in_channels, num_output_channels]
        # kernel: 卷积核
        # 要求是一个Tensor，
        # 具有[filter_height, filter_width, in_channels, out_channels]这样的shape，
        # 具体含义是[卷积核的高度，卷积核的宽度，图像通道数，卷积核个数]，
        # 要求类型与参数input相同，有一个地方需要注意，第三维in_channels，
        # 就是参数input的第四维

        # 返回使用xavier initializer生成的对应shape的variable
        if use_xavier:
            kernel_initializer = tf.keras.initializers.glorot_uniform()
        else:
            kernel_initializer = tf.keras.initializers.TruncatedNormal(stddev=stddev)

        bias_initializer = tf.keras.initializers.Constant(value=initial_bias)

        kernel_regularizer = tf.keras.regularizers.l2(0.)

        stride_h, stride_w = stride

        layer = tf.keras.layers.Conv2D(num_output_channels, 
                                       kernel_size, 
                                       use_bias=True, 
                                       strides=(1, 1), 
                                       padding='valid',
                                       data_format=data_format,
                                       name='conv%d'%(num),
                                       kernel_initializer=kernel_initializer,
                                       bias_initializer=bias_initializer,
                                       kernel_regularizer=kernel_regularizer)

        x = layer(inputs, training=self.is_training)
        if self.bn:
            assert data_format == 'channels_last'
            x = self.batch_normalization(x, training=self.is_training, name="BN%d"%(num))

        if self.activation_fn == "relu":
            layer = tf.keras.layers.ReLU(max_value=None, negative_slope=0.0, threshold=0.0)
            x = layer(x, training=self.is_training)

        return x

    def batch_normalization(self, inputs, name, training):
        layer = tf.keras.layers.BatchNormalization(momentum=0.9, name=name)
        return layer(inputs, training=training)

    def call(self, xyz1, xyz2, points1, points2):
        dist, idx = three_nn(xyz1, xyz2)
        dist = tf.maximum(dist, 1e-10)
        # 将第二维压缩求和
        norm = tf.reduce_sum((1.0/dist),axis=2,keep_dims=True)
        norm = tf.tile(norm,[1,1,3])
        # 根据距离求得点的权重
        weight = (1.0/dist) / norm

        interpolated_points = three_interpolate(points2, idx, weight)

        if points1 is not None:
            new_points1 = tf.concat(axis=2, values=[interpolated_points, points1])  # B,ndataset1,nchannel1+nchannel2
        else:
            new_points1 = interpolated_points
        new_points1 = tf.expand_dims(new_points1, 2)

        for i, num_out_channel in enumerate(self.mlp):
            new_points1 = self.conv2d(new_points1, 
                                      num_out_channel, 
                                      [1,1],
                                      i,
                                      padding='VALID', 
                                      stride=[1,1]
                                      )
        new_points1 = tf.squeeze(new_points1, [2])  # B,ndataset1,mlp[-1]
        return new_points1



@register_conf(name="SA", scope="layer", conf_func="self")
class SALayer(SALayerCore):
    pass

@register_conf(name="FP", scope="layer", conf_func="self")
class FPLayer(FPLayerCore):
    pass