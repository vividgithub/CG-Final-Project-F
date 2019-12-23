# -*- coding: utf-8 -*-
"""
Created on Thu Dec  5 14:15:10 2019

@author: 89341
"""

import numpy as np
import tensorflow as tf
from tf.keras.layers import Dense, Reshape
from tf.keras.layers import Convolution1D, MaxPooling1D, BatchNormalization, Convolution2D
from tf.keras.layers.core import Activation
from tf.keras.layers import Lambda, concatenate
from utils.confutil import register_conf

def mat_mul(A, B):
    return tf.matmul(A, B)

def exp_dim(global_feature, num_points):
    return tf.tile(global_feature, [1, num_points, 1])

#TODO adjust the input shape

@register_conf(name="point-net", scope="layer", conf_func="self")
class PointNet(tf.keras.layers.Layer):
    def __init__(self, num_points=4096, classes=13, label='none', **kwargs):
        super(PointNet, self).__init__(name=label)
        self.num_points = num_points
        self.classes=classes
        self.sub_layers = dict()
        
    def call(self, inputs, **kwargs):
        '''
        Pointnet Architecture
        '''
        # input_Transformation_net
        #input_points = Input(shape=(self.num_points, 3))
        input_points = inputs[..., :3]
        x = Convolution1D(64, 1, activation='relu',
                          input_shape=(self.num_points, 3))(input_points)
        x = BatchNormalization()(x)
        x = Convolution1D(128, 1, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Convolution1D(1024, 1, activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=self.num_points)(x)
        x = Dense(512, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dense(256, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dense(9, weights=[np.zeros([256, 9]), np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32)])(x)
        input_T = Reshape((3, 3))(x)
        
        # forward net
        g = Lambda(mat_mul, arguments={'B': input_T})(input_points)
        g = Convolution1D(64, 1, input_shape=(self.num_points, 3), activation='relu')(g)
        g = BatchNormalization()(g)
        g = Convolution1D(64, 1, input_shape=(self.num_points, 3), activation='relu')(g)
        g = BatchNormalization()(g)
        
        # feature transformation net
        f = Convolution1D(64, 1, activation='relu')(g)
        f = BatchNormalization()(f)
        f = Convolution1D(128, 1, activation='relu')(f)
        f = BatchNormalization()(f)
        f = Convolution1D(1024, 1, activation='relu')(f)
        f = BatchNormalization()(f)
        f = MaxPooling1D(pool_size=self.num_points)(f)
        f = Dense(512, activation='relu')(f)
        f = BatchNormalization()(f)
        f = Dense(256, activation='relu')(f)
        f = BatchNormalization()(f)
        f = Dense(64 * 64, weights=[np.zeros([256, 64 * 64]), np.eye(64).flatten().astype(np.float32)])(f)
        feature_T = Reshape((64, 64))(f)
        
        # forward net
        g = Lambda(mat_mul, arguments={'B': feature_T})(g)
        seg_part1 = g
        g = Convolution1D(64, 1, activation='relu')(g)
        g = BatchNormalization()(g)
        g = Convolution1D(128, 1, activation='relu')(g)
        g = BatchNormalization()(g)
        g = Convolution1D(1024, 1, activation='relu')(g)
        g = BatchNormalization()(g)
        
        # global_feature
        global_feature = MaxPooling1D(pool_size=self.num_points)(g)
        global_feature = Lambda(exp_dim, arguments={'num_points': self.num_points})(global_feature)
        
        # point_net_seg
        c = concatenate([seg_part1, global_feature])
        c = Convolution1D(512, 1, activation='relu')(c)
        c = BatchNormalization()(c)
        c = Convolution1D(256, 1, activation='relu')(c)
        c = BatchNormalization()(c)
        c = Convolution1D(128, 1, activation='relu')(c)
        c = BatchNormalization()(c)
        c = Convolution1D(128, 1, activation='relu')(c)
        c = BatchNormalization()(c)
        prediction = Convolution1D(self.classes, 1, activation='softmax')(c)
        
        return prediction
        '''
        end of pointnet
        '''
#TODO sgpn layer
@register_conf(name="sgpn-output", scope="layer", conf_func="self")
class SGPNOutput(tf.keras.layers.Layer):
    def __init__(self, num_points=4096, classes=13, groups = 50, m=10, label='none', **kwargs):
        #TODO
        super(SGPNOutput, self).__init__(name=label)
        self.num_points = num_points
        self.classes=classes
        self.groups = groups
        self.m=m
        self.sub_layers = dict()
    def call(self, inputs, **kwargs):
        #TODO
        #semantic segmentation
        Fsem = Convolution2D(filters=128, kernel_size=[1,1], strides=[1,1], padding='valid')(inputs)
        ptssemseg_logits = Convolution2D(filters=self.groups, kernel_size=[1,1], strides=[1,1], padding='valid')(Fsem)
        ptssemseg_logits = tf.squeeze(ptssemseg_logits, [2])
        ptssemseg = Activation('softmax')(ptssemseg_logits, name = 'semseg')
        
        #similarity matrix
        batch_size = inputs.get_shape()[0].value
        Fsim = Convolution2D(filters=128, kernel_size=[1,1], strides=[1,1], padding='valid')(inputs)
        Fsim = tf.squeeze(Fsim, [2])
        r = tf.reduce_sum(Fsim * Fsim, 2)
        r = tf.reshape(r, [batch_size, -1, 1])
        print(r.get_shape(),Fsim.get_shape())
        D = r - 2 * tf.matmul(Fsim, tf.transpose(Fsim, perm=[0, 2, 1])) + tf.transpose(r, perm=[0, 2, 1])
        simmat_logits = tf.maximum(self.m * D, 0.)
        
        #confidence map
        Fconf = Convolution2D(filters=128, kernel_size=[1,1], strides=[1,1], padding='valid')(inputs)
        conf_logits = Convolution2D(filters=1, kernel_size=[1,1], strides=[1,1], padding='valid')(Fconf)
        conf_logits = tf.squeeze(conf_logits, [2])
        conf = Activation('sigmoid')(conf_logits)
        
        #return {'semseg': ptssemseg,
         #   'semseg_logits': ptssemseg_logits,
          #  'simmat': simmat_logits,
           # 'conf': conf,
            #'conf_logits': conf_logits}
        return [ptssemseg, ptssemseg_logits, simmat_logits, conf, conf_logits]
        