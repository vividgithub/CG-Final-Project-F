import tensorflow as tf
import os
import sys
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/sampling'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/grouping'))
sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/3d_interpolation'))
from tf_sampling import farthest_point_sample, gather_point
from tf_grouping import query_ball_point, group_point, knn_point
from tf_interpolate import three_nn, three_interpolate
import tensorflow as tf
import legacy.pointfly as pf
from utils.confutil import register_conf
from tensorflow.python.client import device_lib
import tensorflow.keras.backend as K
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

BATCH_SIZE = 16

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
        self.sub_layers = dict()

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
               data_format='channels_last',
               use_xavier=True,
               stddev=1e-3,
               weight_decay=0.0,
               activation_fn="relu",
               is_dist=False,
               training = True,
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
        num_in_channels = inputs.get_shape()[-1]
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
            #kernel_initializer = tf.keras.initializers.glorot_uniform()
            pass
        else:
            #kernel_initializer = tf.keras.initializers.TruncatedNormal(stddev=stddev)
            pass

        #bias_initializer = tf.keras.initializers.Constant(value=initial_bias)

        #kernel_regularizer = tf.keras.regularizers.l2(0.)

        stride_h, stride_w = stride

        layer = self.sub_layers.get('conv%d'%(num))
        if not layer:
            layer = tf.keras.layers.Conv2D(num_output_channels, 
                                        kernel_size, 
                                        use_bias=True, 
                                        strides=(1, 1), 
                                        padding='valid',
                                        data_format=data_format,
                                        name='conv%d'%(num),
                                        kernel_initializer=tf.keras.initializers.glorot_uniform(),
                                        bias_initializer=tf.keras.initializers.Constant(value=initial_bias),
                                        kernel_regularizer=tf.keras.regularizers.l2(0.))
            self.sub_layers['conv%d'%(num)] = layer
        
        x = layer(inputs, training=training)
        if self.bn:
            assert data_format == 'channels_last'
            x = self.batch_normalization(x, training=training, name="BN%d"%(num))

        if activation_fn == "relu":
            layer = self.sub_layers.get('relu%d'%(num))
            if not layer:
                layer = tf.keras.layers.ReLU(max_value=None, negative_slope=0.0, threshold=0.0)
                self.sub_layers['relu%d'%(num)] = layer
            x = layer(x, training=training)

        return x

    def batch_normalization(self, inputs, name, training):
        layer = self.sub_layers.get(name)
        if not layer:
            layer = tf.keras.layers.BatchNormalization(momentum=0.9, name=name)
            self.sub_layers[name] = layer
        return layer(inputs, training=training)

    def call(self, xyz, points, training, **kwargs):
        data_format = 'channels_first' if self.use_nchw else 'channels_last'

        new_xyz, new_points, idx, grouped_xyz = self.sample_and_group(xyz, points)


        for i, num_out_channel in enumerate(self.mlp):
            new_points = self.conv2d(new_points, 
                                     num_out_channel, 
                                     [1,1],
                                     i,
                                     padding='VALID', 
                                     stride=[1,1],
                                     data_format=data_format,
                                     training = training)

        # Pooling in Local Regions
        # pooling = max
        if self.pooling=='max':
            new_points = tf.math.reduce_max(new_points, axis=[2], keepdims=True, name='maxpool')

        new_points = tf.squeeze(new_points, [2])  # (batch_size, npoints, mlp2[-1])
        return new_xyz, new_points, idx


class FPLayerCore(tf.keras.layers.Layer):
    """
    The X-Conv kernel used in "PointCNN"(https://arxiv.org/abs/1801.07791)
    """
    def __init__(self, mlp, is_training, bn=True, label=None, **kwargs):
        super(FPLayerCore, self).__init__(name=label)
        self.mlp = mlp
        self.is_training = is_training
        self.bn = bn
        self.sub_layers = dict()

    def conv2d(self, 
               inputs,
               num_output_channels,
               kernel_size,
               num,
               stride=[1, 1],
               padding='SAME',
               data_format='channels_last',
               use_xavier=True,
               stddev=1e-3,
               weight_decay=0.0,
               activation_fn="relu",
               is_dist=False,
               initial_bias=0.0,
               training = True
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
        num_in_channels = inputs.get_shape()[-1]
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

        layer = self.sub_layers.get('conv%d'%(num))
        if not layer:
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
            self.sub_layers['conv%d'%(num)] = layer

        x = layer(inputs, training=training)
        if self.bn:
            assert data_format == 'channels_last'
            x = self.batch_normalization(x, training=training, name="BN%d"%(num))

        if activation_fn == "relu":
            layer = self.sub_layers.get('relu%d'%(num))
            if not layer:
                layer = tf.keras.layers.ReLU(max_value=None, negative_slope=0.0, threshold=0.0)
                self.sub_layers['relu%d'%(num)] = layer
            x = layer(x, training=training)

        return x

    def batch_normalization(self, inputs, name, training):
        layer = self.sub_layers.get(name)
        if not layer:
            layer = tf.keras.layers.BatchNormalization(momentum=0.9, name=name)
            self.sub_layers[name] = layer
        return layer(inputs, training=training)

    def call(self, xyz1, xyz2, points1, points2, training):
        dist, idx = three_nn(xyz1, xyz2)
        dist = tf.maximum(dist, 1e-10)
        # 将第二维压缩求和
        norm = tf.math.reduce_sum((1.0/dist),axis=2,keepdims=True)
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
                                      stride=[1,1],
                                      training = training
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
        is_dist=False,
        **kwargs
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
        self.sub_layers = dict()
        self.num = 0
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

    def conv1d(self, inputs, training = True):
        #self.num = self.num + 1
        layer = self.sub_layers.get("conv1d%d"%(self.num))
        if not layer:
            layer = tf.keras.layers.Conv1D(
                    self.num_output_channels, 
                    self.kernel_size, 
                    strides = self.stride, 
                    padding = self.padding, 
                    kernel_initializer = tf.keras.initializers.TruncatedNormal(stddev = self.stddev),
                    bias_initializer = tf.keras.initializers.Zeros(),
                    kernel_regularizer = tf.keras.regularizers.l2(0.),
                )
            self.sub_layers["conv1d%d"%(self.num)] = layer

        
        x = layer(inputs, training = training)
        return x

    def batch_normalization(self, inputs, training):
        #self.num = self.num + 1
        layer = self.sub_layers.get("batch_conv1d%d"%(self.num))
        if not layer:
            layer = tf.keras.layers.BatchNormalization(
                momentum = self.bn_decay,
                beta_initializer = tf.keras.initializers.Zeros(),
                gamma_initializer = tf.keras.initializers.Ones()
            )
            self.sub_layers["batch_conv1d%d"%(self.num)] = layer

        if self.bn:
            outputs = layer(inputs, training = training)

        return outputs

    def activation(self, inputs):
        #self.num = self.num + 1
        layer = self.sub_layers.get("activation%d"%(self.num))
        if not layer:
            layer = tf.keras.layers.Activation(self.activation_fn)
            self.sub_layers["activation%d"%(self.num)] = layer

        outputs = layer(inputs)
        return outputs

    #inputs: 3-D tensor BxLxC
    def call(self, inputs, training):
        x = self.conv1d(inputs, training = training)
        x1 = self.batch_normalization(x, training = training)
        outputs = self.activation(x1)
        
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
        self.sub_layers = dict()
        self.num = 0
    
    def call(self, inputs, training):
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
        self.num = self.num + 1
        layer = self.sub_layers.get("dropout%d"%(self.num))
        if not layer:
            layer = tf.keras.layers.Dropout(rate = self.keep_prob, noise_shape = self.noise_shape)
            self.sub_layers["dropout%d"%(self.num)] = layer
        if training:
            outputs = layer(inputs, training = training)
        else:
            outputs = inputs
        return outputs

@register_conf(name="pairwise-distance-l1", scope="layer", conf_func="self")
class PairWiseDistanceL1Layer(tf.keras.layers.Layer):
    def __init__(self, label = None, batch_size = BATCH_SIZE, **kwargs): #24
        super(PairWiseDistanceL1Layer, self).__init__(name=label)
        self.batch_size = batch_size

    def call(self, inputs):
        point_cloud = inputs
        og_batch_size = self.batch_size
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
        batch_size = BATCH_SIZE, #24
        **kwargs
        ):
        super(KnnThres, self).__init__(name = label)
        self.k = k
        self.batch_size = batch_size
        self.thres = thres

    def call(self, inputs):
        """Get KNN based on the pairwise distance.
        Args:
            pairwise distance: (batch_size, num_points, num_points)
            k: int

        Returns:
            nearest neighbors: (batch_size, num_points, k)
        """
        og_batch_size = self.batch_size
        neg_adj = -inputs
        vals, nn_idx = tf.math.top_k(neg_adj, k=self.k)

        to_add = tf.range(4096)
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
        batch_size = BATCH_SIZE #24
        num_points = 4096
        num_dims = 128

        idx_ = tf.range(batch_size) * num_points
        idx_ = tf.reshape(idx_, [batch_size, 1, 1]) 

        point_cloud_flat = tf.reshape(point_cloud, [-1, num_dims])
        point_cloud_neighbors = tf.gather(point_cloud_flat, nn_idx+idx_)

        edge_feature = tf.math.reduce_max(point_cloud_neighbors, axis = -2, keepdims = False)

        return edge_feature

@register_conf(name="loss-layer", scope="layer", conf_func="self")
class LossLayer(tf.keras.layers.Layer):
    def __init__(
        self,
        is_training = True,
        label = None,
        **kwargs
    ):
        super(LossLayer, self).__init__(name = label)
        self.is_training = is_training
        self.segmented_sum = []
        self.m = tf.keras.metrics.SparseCategoricalAccuracy()

    def discriminative_loss(self, prediction, correct_label, feature_dim,
                            delta_v, delta_d, param_var, param_dist, param_reg):
        ''' Iterate over a batch of prediction/label and cumulate loss
        :return: discriminative loss and its three components
        '''

        # i: 第i个batch, i >= B时循环停止
        def cond(label, batch, out_loss, out_var, out_dist, out_reg, i):
            return tf.less(i, tf.shape(batch)[0])

        def body(label, batch, out_loss, out_var, out_dist, out_reg, i):
            disc_loss, l_var, l_dist, l_reg = self.discriminative_loss_single(prediction[i], correct_label[i], feature_dim,
                                                                        delta_v, delta_d, param_var, param_dist, param_reg)
            # 在第i个index下写进后面的value
            out_loss = out_loss.write(i, disc_loss)
            out_var = out_var.write(i, l_var)
            out_dist = out_dist.write(i, l_dist)
            out_reg = out_reg.write(i, l_reg)

            return label, batch, out_loss, out_var, out_dist, out_reg, i + 1

        # TensorArray is a data structure that support dynamic writing
        output_ta_loss = tf.TensorArray(dtype=tf.float32,
                                        size=0,
                                        dynamic_size=True)
        output_ta_var = tf.TensorArray(dtype=tf.float32,
                                    size=0,
                                    dynamic_size=True)
        output_ta_dist = tf.TensorArray(dtype=tf.float32,
                                        size=0,
                                        dynamic_size=True)
        output_ta_reg = tf.TensorArray(dtype=tf.float32,
                                    size=0,
                                    dynamic_size=True)

        _, _, out_loss_op, out_var_op, out_dist_op, out_reg_op, _ = tf.while_loop(cond, body, [correct_label,
                                                                                            prediction,
                                                                                            output_ta_loss,
                                                                                            output_ta_var,
                                                                                            output_ta_dist,
                                                                                            output_ta_reg,
                                                                                            0])
        # 将array的元素堆叠成tensor
        out_loss_op = out_loss_op.stack()
        out_var_op = out_var_op.stack()
        out_dist_op = out_dist_op.stack()
        out_reg_op = out_reg_op.stack()

        disc_loss = tf.reduce_mean(out_loss_op)
        l_var = tf.reduce_mean(out_var_op)
        l_dist = tf.reduce_mean(out_dist_op)
        l_reg = tf.reduce_mean(out_reg_op)

        return disc_loss, l_var, l_dist, l_reg

    def discriminative_loss_single(self, prediction, correct_label, feature_dim,
                                delta_v, delta_d, param_var, param_dist, param_reg):
        ''' Discriminative loss for a single prediction/label pair.
        :param prediction: inference of network
        :param correct_label: instance label
        :feature_dim: feature dimension of prediction
        :param label_shape: shape of label
        :param delta_v: cutoff variance distance
        :param delta_d: curoff cluster distance
        :param param_var: weight for intra cluster variance
        :param param_dist: weight for inter cluster distances
        :param param_reg: weight regularization
        '''

        ### Reshape so pixels are aligned along a vector
        #correct_label = tf.reshape(correct_label, [label_shape[1] * label_shape[0]])
        reshaped_pred = tf.reshape(prediction, [-1, feature_dim])

        ### Count instances
        
        #tf.enable_eager_execution()
        try:
            unique_labels, unique_id, counts = tf.stop_gradient(tf.unique_with_counts(correct_label))
        except:
            unique_labels, unique_id, counts = tf.unique_with_counts(correct_label)
        #tf.no_gradient('UniqueWithCounts')

        counts = tf.cast(counts, tf.int32)
        num_instances = tf.size(unique_labels)
        
        #segmented_sum = []
        def cond(label, unique_id, reshaped_pred, out_segmented_sum, i):
            return tf.less(i, tf.shape(label)[0])

        def body(label, unique_id, reshaped_pred, out_segmented_sum, i):
            bool_array = tf.math.equal(unique_id, i)
            bool_array = tf.reshape(bool_array, (4096,))
            #try:
            selected_pred = tf.boolean_mask(reshaped_pred, bool_array)
            #except:
            #selected_pred = tf.ones([1])
            selected_sum_pred = tf.math.reduce_sum(selected_pred, axis = 0)
            out_segmented_sum = out_segmented_sum.write(i, selected_sum_pred)

            return label, unique_id, reshaped_pred, out_segmented_sum, i + 1

        segmented_sum = tf.TensorArray(dtype=tf.float32,
                                        size=0,
                                        dynamic_size=True)

        _, _, _, segmented_sum, _ = tf.while_loop(cond, body, [unique_labels, unique_id, reshaped_pred, segmented_sum, 0])

        segmented_sum = segmented_sum.stack()
        #segmented_sum_1 = tf.stack(segmented_sum)
        #segmented_sum = tf.compat.v2.math.unsorted_segment_sum(reshaped_pred, unique_id, num_instances)

        counts = tf.cast(counts, dtype = tf.float32)
        mu = tf.math.divide(segmented_sum, tf.reshape(counts, (-1, 1)))
        mu_expand = tf.gather(mu, unique_id)

        ### Calculate l_var
        #distance = tf.norm(tf.subtract(mu_expand, reshaped_pred), axis=1)
        #tmp_distance = tf.subtract(reshaped_pred, mu_expand)
        tmp_distance = reshaped_pred - mu_expand
        distance = tf.norm(tmp_distance, ord=1, axis=1)

        distance = tf.subtract(distance, delta_v)
        distance = tf.clip_by_value(distance, 0., distance)
        distance = tf.square(distance)
        
        l_var = tf.TensorArray(dtype=tf.float32,
                                        size=0,
                                        dynamic_size=True)

        l_var = tf.TensorArray(dtype=tf.float32,
                                        size=0,
                                        dynamic_size=True)

        _, _, _, l_var, _ = tf.while_loop(cond, body, [unique_labels, unique_id, distance, l_var, 0])

        '''
        if unique_labels.get_shape()[0] is None:
            len = 0
        else:
            len = unique_labels.get_shape()[0]
        for i in range(len):
            bool_array = tf.math.equal(unique_id, i)
            bool_array = tf.reshape(bool_array, (4096, ))
            selected_dist = tf.boolean_mask(distance, bool_array)
            selected_sum_dist = tf.math.reduce_sum(selected_dist, axis = 0)
            l_var = l_var.write(i, selected_sum_dist)
        '''

        l_var = l_var.stack()

        #l_var = tf.compat.v2.math.unsorted_segment_sum(distance, unique_id, num_instances)
        l_var = tf.math.divide(l_var, counts)
        l_var = tf.reduce_sum(l_var)
        l_var = tf.math.divide(l_var, tf.cast(num_instances, tf.float32))

        ### Calculate l_dist

        # Get distance for each pair of clusters like this:
        #   mu_1 - mu_1
        #   mu_2 - mu_1
        #   mu_3 - mu_1
        #   mu_1 - mu_2
        #   mu_2 - mu_2
        #   mu_3 - mu_2
        #   mu_1 - mu_3
        #   mu_2 - mu_3
        #   mu_3 - mu_3

        mu_interleaved_rep = tf.tile(mu, [num_instances, 1])
        mu_band_rep = tf.tile(mu, [1, num_instances])
        mu_band_rep = tf.reshape(mu_band_rep, (num_instances * num_instances, feature_dim))

        mu_diff = tf.subtract(mu_band_rep, mu_interleaved_rep)

        # Filter out zeros from same cluster subtraction
        eye = tf.eye(num_instances)
        zero = tf.zeros(1, dtype=tf.float32)
        diff_cluster_mask = tf.equal(eye, zero)
        diff_cluster_mask = tf.reshape(diff_cluster_mask, [-1])
        mu_diff_bool = tf.boolean_mask(mu_diff, diff_cluster_mask)

        #intermediate_tensor = tf.reduce_sum(tf.abs(mu_diff),axis=1)
        #zero_vector = tf.zeros(1, dtype=tf.float32)
        #bool_mask = tf.not_equal(intermediate_tensor, zero_vector)
        #mu_diff_bool = tf.boolean_mask(mu_diff, bool_mask)

        mu_norm = tf.norm(mu_diff_bool, ord=1, axis=1)
        mu_norm = tf.subtract(2. * delta_d, mu_norm)
        mu_norm = tf.clip_by_value(mu_norm, 0., mu_norm)
        mu_norm = tf.square(mu_norm)

        l_dist = tf.reduce_mean(mu_norm)

        def rt_0(): return 0.
        def rt_l_dist(): return l_dist
        l_dist = tf.cond(tf.equal(1, num_instances), rt_0, rt_l_dist)
        
        ### Calculate l_reg
        l_reg = tf.reduce_mean(tf.norm(mu, ord=1, axis=1))

        param_scale = 1.
        l_var = param_var * l_var
        l_dist = param_dist * l_dist
        l_reg = param_reg * l_reg

        loss = param_scale * (l_var + l_dist + l_reg)

        return loss, l_var, l_dist, l_reg

    def call(self, pred_ins, pred_sem, true_ins, true_sem):
        y_true = true_sem
        y_pred = pred_sem
        
        true_ins = tf.squeeze(true_ins)
        true_sem = tf.squeeze(true_sem)

        classify_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(true_sem, pred_sem)
        # discriminative loss
        feature_dim = pred_ins.get_shape()[-1]
        delta_v = 0.5
        delta_d = 1.5
        param_var = 1.
        param_dist = 1.
        param_reg = 0.001
        
        disc_loss = 0
        # 返回discriminative loss以及附带的三个参数
        disc_loss, l_var, l_dist, l_reg = self.discriminative_loss(pred_ins, true_ins, feature_dim,
                                             delta_v, delta_d, param_var, param_dist, param_reg)

        # total loss
        loss = classify_loss + disc_loss

        self.add_loss(loss, inputs = True)
        self.m.update_state(y_true, y_pred)
        accuracy = self.m.result()
        #print('Final result: ', m.result().numpy())  # Final result: 0.5

        self.add_metric(accuracy, aggregation="mean", name = "accuracy")
        
        return y_pred

