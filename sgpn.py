# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 14:15:10 2019

@author: lznaf
"""

import tensorflow as tf
from tensorflow.keras.layers import Dense, Reshape, Flatten
from tensorflow.keras.layers import BatchNormalization, Conv2D
from tensorflow.keras.layers import Activation, MaxPool2D
from utils.confutil import register_conf

def mat_mul(A, B):
    return tf.matmul(A, B)

def exp_dim(global_feature, num_points):
    return tf.tile(global_feature, [1, num_points, 1])

#@register_conf(name="pointNet", scope="layer", conf_func="self")
def PointNet(inputs, num_points=4096, classes=13, label='pointNet', **kwargs):
    '''
    Pointnet Architecture
    '''
    point_cloud = inputs[0]
    num_point = num_points

    input_image = tf.expand_dims(point_cloud, -1)
    # CONV
    net = Conv2D(filters=64, kernel_size=[1, 9], strides=[1, 1], padding='valid', name = 'conv1')(input_image)
    net = BatchNormalization(name='b1')(net)
    net = Conv2D(filters=64, kernel_size=[1, 1], strides=[1, 1], padding='valid', name = 'conv2')(net)
    net = BatchNormalization(name='b2')(net)
    net = Conv2D(filters=64, kernel_size=[1, 1], strides=[1, 1], padding='valid', name = 'conv3')(net)
    net = BatchNormalization(name='b3')(net)
    net = Conv2D(filters=128, kernel_size=[1, 1], strides=[1, 1], padding='valid', name = 'conv4')(net)
    net = BatchNormalization(name='b4')(net)
    points_feat1 = Conv2D(filters=1024, kernel_size=[1, 1], strides=[1, 1], padding='valid', name='conv5')(net)
    net = BatchNormalization(name='b5')(net)
    # MAX
    pc_feat1 = MaxPool2D(pool_size=[num_point, 1], strides = [2,2], padding = 'valid', name='pool1')(points_feat1)
    # FC
    pc_feat1 = Flatten(name='f1')(pc_feat1)
    pc_feat1 = Dense(256, name='d1')(pc_feat1)
    pc_feat1 = BatchNormalization(name='b6')(pc_feat1)
    pc_feat1 = Dense(128, name='d2')(pc_feat1)
    pc_feat1 = BatchNormalization(name='b7')(pc_feat1)

    # CONCAT
    pc_feat1 = Reshape((1, 1, 128))(pc_feat1)
    pc_feat1_expand = tf.tile(pc_feat1, (1, num_point, 1, 1))
    points_feat1_concat = tf.concat(axis=3, values=[points_feat1, pc_feat1_expand])

    # CONV
    net = Conv2D(filters=512, kernel_size=[1, 1], strides=[1, 1], padding='valid', name='conv6')(points_feat1_concat)
    net = BatchNormalization(name='b8')(net)
    net = Conv2D(filters=256, kernel_size=[1, 1], strides=[1, 1], padding='valid', name='conv7')(net)
    net = BatchNormalization(name='b9')(net)

    return net
    '''
    end of pointnet
    '''
#@register_conf(name="sgpn-output", scope="layer", conf_func="self")
def SGPNOutput(inputs, num_points=4096, classes=13, groups = 50, m=10, label='sgpnOut', **kwargs):
    #semantic segmentation
    Fsem = Conv2D(filters=128, kernel_size=[1,1], strides=[1,1], padding='valid', name='conv8')(inputs)
    ptssemseg_logits = Conv2D(filters=groups, kernel_size=[1,1], strides=[1,1], padding='valid', name='conv9')(Fsem)
    ptssemseg_logits = tf.squeeze(ptssemseg_logits, [2])
    ptssemseg = Activation('softmax', name='a1')(ptssemseg_logits)
    
    #similarity matrix
    Fsim = Conv2D(filters=128, kernel_size=[1,1], strides=[1,1], padding='valid', name='conv10')(inputs)
    Fsim = tf.squeeze(Fsim, [2])
    r = tf.reduce_sum(Fsim * Fsim, 2)
    print(r.shape)
    r = Reshape([-1, 1], name='r2')(r)
    print(r.get_shape(),Fsim.get_shape())
    D = r - 2 * tf.matmul(Fsim, tf.transpose(Fsim, perm=[0, 2, 1])) + tf.transpose(r, perm=[0, 2, 1])
    simmat_logits = tf.maximum(m * D, 0.)
    
    #confidence map
    Fconf = Conv2D(filters=128, kernel_size=[1,1], strides=[1,1], padding='valid', name='conv11')(inputs)
    conf_logits = Conv2D(filters=1, kernel_size=[1,1], strides=[1,1], padding='valid', name='conv12')(Fconf)
    conf_logits = tf.squeeze(conf_logits, [2])
    conf = Activation('sigmoid', name='a2')(conf_logits)
    
    #return {'semseg': ptssemseg,
    #'semseg_logits': ptssemseg_logits,
    #'simmat': simmat_logits,
    #'conf': conf,
    #'conf_logits': conf_logits}
    return [ptssemseg, ptssemseg_logits, simmat_logits, conf, conf_logits]
        