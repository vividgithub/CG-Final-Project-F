import tensorflow as tf
from tf_sampling import farthest_point_sample, gather_point
from tf_grouping import query_ball_point, group_point, knn_point
from tf_interpolate import three_nn, three_interpolate
import tensorflow as tf
import legacy.pointfly as pf
from utils.confutil import register_conf

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

@register_conf(name="conv-1d", scope="layer", conf_func="self")
class Conv1dLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        num_output_channels,
        kernel_size,
        stride=1,
        padding='SAME',
        use_xavier=True,
        stddev=1e-3,
        weight_decay=0.0,
        activation_fn=tf.keras.activations.relu,
        bn=False,
        bn_decay=None,
        is_training=None,
        is_dist=False
    ):
        super(Conv1dLayer, self).__init__()
        self.num_output_channels = num_output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_xavier = use_xavier
        self.stddev = stddev
        self.weight_decay = weight_decay
        self.activation_fn = activation_fn
        self.bn = bn
        self.bn_decay = bn_decay
        self.is_training = is_training
        self.is_dist = is_dist
        """ 1D convolution with non-linear operation.

        Args:
            num_output_channels: int
            kernel_size: int
            stride: int
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

    #inputs: 3-D tensor BxLxC
    def call(self, inputs):
        layer_conv1d = tf.keras.layers.Conv1d(
            self.num_output_channels, 
            self.kernel_size, 
            strides = self.stride, 
            padding = self.padding, 
            kernel_initializer = tf.keras.initializers.TruncatedNormal(stddev = self.stddev),
            bias_initializer = tf.keras.initializers.Zeros(),
            kernel_regularizer = tf.keras.regularizers.l2(0.)
        )
        x = layer_conv1d(inputs, training = self.is_training)

        layer_batch_norm_for_conv1d = tf.keras.layers.BatchNormalization(
            momentum = self.bn_decay,
            beta_initializer = tf.keras.initializers.Zeros(),
            gamma_initializer = tf.keras.initializers.Ones()
        )

        if self.bn:
            outputs = layer_batch_norm_for_conv1d(x, training = self.is_training)

        layer_activation = tf.keras.layers.Activation(self.activation_fn)

        outputs = layer_activation(outputs)

        return outputs

@register_conf(name="dropout", scope="layer", conf_func="self")
class DropOutLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        is_training,
        keep_prob = 0.5,
        noise_shape = None,
        label = None,
        **kwargs
    ):
        super(DropOutLayer, self).__init__(name=label)
        self.is_training = is_training
        self.keep_prob = keep_prob
        self.noise_shape = noise_shape
    
    def call(self, inputs):
        """ Dropout layer.

        Args:
            inputs: tensor
            is_training: boolean tf.Variable
            scope: string
            keep_prob: float in [0,1]
            noise_shape: list of ints

        Returns:
            tensor variable
        """
        drop_out_layer = tf.keras.Dropout(rate = self.keep_prob, noise_shape = self.noise_shape)
        if self.is_training:
            outputs = drop_out_layer(inputs, training = self.is_training)
        else
            outputs = inputs
        return outputs

@register_conf(name="pairwise-distance-l1", scope="layer", conf_func="self")
class PairWiseDistanceL1Layer(tf.keras.layers.Layer):
    def __init__(self, label = None, **kwargs):
        super(PairWiseDistanceL1Layer, self).__init__(name=label)

    def call(self, inputs):
        og_batch_size = point_cloud.get_shape().as_list()[0]
        point_cloud = tf.squeeze(point_cloud)
        if og_batch_size == 1:
            point_cloud = tf.expand_dims(point_cloud, 0)
        pairwise_distance = []
        for idx in range(og_batch_size):
            idx_point_cloud = point_cloud[idx, :, :]
            l1 = tf.reduce_sum(tf.abs(tf.subtract(idx_point_cloud, tf.expand_dims(idx_point_cloud, 1))), 
                                axis=2)
            pairwise_distance.append(l1)
        return tf.stack(pairwise_distance, axis=0)

@register_conf(name="knn-thres", scope="layer", conf_func="self")
class KnnThres(tf.keras.layers.Layer):
    def __init__(
        self,
        k = 20,
        thres = 0.5,
        label = None,
        **kwargs
        ):
        super(KnnThres, self).__init__(name = label)
        self.k = k
        self.thres = thres

    def call(self, inputs):
        """Get KNN based on the pairwise distance.
        Args:
            pairwise distance: (batch_size, num_points, num_points)
            k: int

        Returns:
            nearest neighbors: (batch_size, num_points, k)
        """
        og_batch_size = inputs.get_shape().as_list()[0]
        neg_adj = -inputs
        vals, nn_idx = tf.math.top_k(neg_adj, k=self.k)

        to_add = tf.range(nn_idx.get_shape()[1])
        to_add = tf.reshape(to_add, [-1, 1])
        to_add = tf.tile(to_add, [1, self.k]) #[N,k]

        final_nn_idx = []
        for idx in range(og_batch_size):
            idx_vals = vals[idx, :, :]
            idx_nn_idx = nn_idx[idx, :, :]
            mask = tf.cast(idx_vals < -1*self.thres, tf.int32) # [N, K]
            idx_to_add = to_add * mask
            idx_nn_idx = idx_nn_idx * (1 - mask) + idx_to_add 
            final_nn_idx.append(idx_nn_idx)
        
        nn_idx = tf.stack(final_nn_idx, axis=0)

        return tf.stop_gradient(nn_idx)

@register_conf(name="get-local-feature", scope="layer", conf_func="self")
class GetLocalFeature(tf.keras.layers.Layer):
    def __init__(
        self,
        k,
        label = None,
        **kwargs
    ):
        super(GetLocalFeature, self).__init__(name = label)
        self.k = k

    def call(self, input1, input2):
        point_cloud = input1
        nn_idx = input2
        """Construct edge feature for each point
        Args:
            point_cloud: (batch_size, num_points, 1, num_dims)
            nn_idx: (batch_size, num_points, k)
            k: int

        Returns:
            edge features: (batch_size, num_points, k, num_dims)
        """
        og_batch_size = point_cloud.get_shape().as_list()[0]
        point_cloud = tf.squeeze(point_cloud)
        if og_batch_size == 1:
            point_cloud = tf.expand_dims(point_cloud, 0)

        point_cloud_central = point_cloud

        point_cloud_shape = point_cloud.get_shape()
        batch_size = point_cloud_shape[0]
        num_points = point_cloud_shape[1]
        num_dims = point_cloud_shape[2]

        idx_ = tf.range(batch_size) * num_points
        idx_ = tf.reshape(idx_, [batch_size, 1, 1]) 

        point_cloud_flat = tf.reshape(point_cloud, [-1, num_dims])
        point_cloud_neighbors = tf.gather(point_cloud_flat, nn_idx+idx_)

        edge_feature = tf.reduce_max(point_cloud_neighbors, axis = -2, keep_dims = False)

        return edge_feature