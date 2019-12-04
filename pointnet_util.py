import tensorflow as tf
from time import time
import numpy as np

def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

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
    print(src.shape)
    print(dst.shape)
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dst2=tf.keras.backend.permute_dimensions(dst,(0, 2, 1))
    dist = -2 * tf.matmul(src, dst2)
    print(dist.shape)
    tmp=tf.reduce_sum(src ** 2, -1)
    print(tmp.shape)
    tmp=tf.reshape(tmp, (B, N, 1))
    dist += tmp
    print(dist.shape)
    tmp=tf.reduce_sum(dst ** 2, -1)
    print(tmp.shape)
    tmp=tf.reshape(tmp, (B, 1, M))
    dist += tmp
    return dist


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = tf.range(0, B, dtype=tf.int64)
    batch_indices = tf.reshape(batch_indices, view_shape)
    batch_indices = tf.tile(batch_indices, repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, C]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    B, N, C = xyz.shape
    centroids = tf.zeros((B, npoint), dtype=tf.int64)

    distance = tf.ones((B, N),dtype=tf.float64) * 1e10
    farthest = tf.random.uniform((B,),minval=0, maxval=N, dtype=tf.int64)
    batch_indices = tf.range(0, B, dtype=tf.int64)
    centroids=centroids.numpy()
    distance=distance.numpy()
    for i in range(npoint):
        centroids[:, i]=farthest
        centroid = tf.reshape(xyz[batch_indices, farthest, :],(B, 1, 3))
        dist = tf.reduce_sum((xyz - centroid) ** 2, -1)
        dist=dist.numpy()
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = tf.argmax(distance, -1)
    return tf.convert_to_tensor(centroids)


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = tf.range(0, N, dtype=tf.int64)
    group_idx = tf.reshape(group_idx, (1, 1, N))
    group_idx = tf.tile(group_idx, [B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    mask=sqrdists.numpy() > radius ** 2
    #mask=mask.numpy().astype(np.int16)
    #print('###')
    #print(mask.shape)
    #print(N)
    #print(group_idx.shape)
    group_idx=group_idx.numpy()
    #print(mask)
    group_idx[mask] = N
    group_idx = tf.sort(group_idx,-1)[:, :, :nsample]
    print(group_idx)
    group_first = group_idx[:, :, 0]
    group_first = tf.reshape(group_first, (B, S, 1))
    group_first = tf.tile(group_first, [1, 1, nsample])
    group_idx=group_idx.numpy()
    mask =  group_idx== N
    group_first=group_first.numpy()
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group_all(xyz, points, use_xyz=True):
    """
    Input:
        xyz: input points position data, [B, N, C]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, C]
        new_points: sampled points data, [B, 1, N, C+D]
    """
    batch_size = xyz.get_shape()[0].value
    nsample = xyz.get_shape()[1].value
    new_xyz = tf.constant(np.tile(np.array([0,0,0]).reshape((1,1,3)), (batch_size,1,1)),dtype=tf.float32) # (batch_size, 1, 3)
    idx = tf.constant(np.tile(np.array(range(nsample)).reshape((1,1,nsample)), (batch_size,1,1)))
    grouped_xyz = tf.reshape(xyz, (batch_size, 1, nsample, 3)) # (batch_size, npoint=1, nsample, 3)
    if points is not None:
        if use_xyz:
            new_points = tf.concat([xyz, points], axis=2) # (batch_size, 16, 259)
        else:
            new_points = points
        new_points = tf.expand_dims(new_points, 1) # (batch_size, 1, 16, 259)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points, idx, grouped_xyz
#    B, N, C = xyz.shape

#    new_xyz = tf.zeros((B, 1, C))
#    grouped_xyz = tf.reshape(xyz, (B, 1, N, C))
#    if points is not None:
#        new_points = tf.concat([grouped_xyz, tf.reshape(points,(B, 1, N, -1))], dim=-1)
#    else:
#        new_points = grouped_xyz
#    return new_xyz, new_points

def sample_and_group(npoint, radius, nsample, xyz, points, knn=False, use_xyz=True,returnfps=True):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, C]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, C]
        new_points: sampled points data, [B, 1, N, C+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint, C]
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    new_xyz=tf.reshape(new_xyz, (B, S, 1, C))
    grouped_xyz_norm = grouped_xyz - new_xyz
    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = tf.concat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, fps_idx, grouped_xyz
    else:
        return new_xyz, new_points

def pointnet_sa_module(xyz, points, npoint, radius, nsample, mlp, mlp2, group_all, is_training, bn_decay, scope, bn=True, pooling='max', knn=False, use_xyz=True, use_nchw=False):
    ''' PointNet Set Abstraction (SA) Module
        Input:
            xyz: (batch_size, ndataset, 3) TF tensor
            points: (batch_size, ndataset, channel) TF tensor
            npoint: int32 -- #points sampled in farthest point sampling
            radius: float32 -- search radius in local region
            nsample: int32 -- how many points in each local region
            mlp: list of int32 -- output size for MLP on each point
            mlp2: list of int32 -- output size for MLP on each region
            group_all: bool -- group all points into one PC if set true, OVERRIDE
                npoint, radius and nsample settings
            use_xyz: bool, if True concat XYZ with local point features, otherwise just use point features
            use_nchw: bool, if True, use NCHW data format for conv2d, which is usually faster than NHWC format
        Return:
            new_xyz: (batch_size, npoint, 3) TF tensor
            new_points: (batch_size, npoint, mlp[-1] or mlp2[-1]) TF tensor
            idx: (batch_size, npoint, nsample) int32 -- indices for local regions
    '''
    data_format = 'NCHW' if use_nchw else 'NHWC'

    # Sample and Grouping
    if group_all:
        nsample = xyz.get_shape()[1].value
        new_xyz, new_points, idx, grouped_xyz = sample_and_group_all(xyz, points, use_xyz)
    else:
        new_xyz, new_points, idx, grouped_xyz = sample_and_group(npoint, radius, nsample, xyz, points, knn, use_xyz)

    # Point Feature Embedding
    if use_nchw: new_points = tf.transpose(new_points, [0,3,1,2])
    for i, num_out_channel in enumerate(mlp):
        new_points = tf_util.conv2d(new_points, num_out_channel, [1,1],
                                    padding='VALID', stride=[1,1],
                                    bn=bn, is_training=is_training,
                                    scope='conv%d'%(i), bn_decay=bn_decay,
                                    data_format=data_format)
    if use_nchw: new_points = tf.transpose(new_points, [0,2,3,1])

    # Pooling in Local Regions
    if pooling=='max':
        new_points = tf.reduce_max(new_points, axis=[2], keep_dims=True, name='maxpool')
    elif pooling=='avg':
        new_points = tf.reduce_mean(new_points, axis=[2], keep_dims=True, name='avgpool')
    elif pooling=='weighted_avg':
        with tf.variable_scope('weighted_avg'):
            dists = tf.norm(grouped_xyz,axis=-1,ord=2,keep_dims=True)
            exp_dists = tf.exp(-dists * 5)
            weights = exp_dists/tf.reduce_sum(exp_dists,axis=2,keep_dims=True) # (batch_size, npoint, nsample, 1)
            new_points *= weights # (batch_size, npoint, nsample, mlp[-1])
            new_points = tf.reduce_sum(new_points, axis=2, keep_dims=True)
    elif pooling=='max_and_avg':
        max_points = tf.reduce_max(new_points, axis=[2], keep_dims=True, name='maxpool')
        avg_points = tf.reduce_mean(new_points, axis=[2], keep_dims=True, name='avgpool')
        new_points = tf.concat([avg_points, max_points], axis=-1)

    # [Optional] Further Processing
    if mlp2 is not None:
        if use_nchw: new_points = tf.transpose(new_points, [0,3,1,2])
        for i, num_out_channel in enumerate(mlp2):
            new_points = tf_util.conv2d(new_points, num_out_channel, [1,1],
                                        padding='VALID', stride=[1,1],
                                        bn=bn, is_training=is_training,
                                        scope='conv_post_%d'%(i), bn_decay=bn_decay,
                                        data_format=data_format)
        if use_nchw: new_points = tf.transpose(new_points, [0,2,3,1])

    new_points = tf.squeeze(new_points, [2]) # (batch_size, npoints, mlp2[-1])
    return new_xyz, new_points, idx


