# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 14:36:57 2019

@author: 89341
"""

import tensorflow as tf
import numpy as np



def SGPNloss(labels:list , net_output:list):
    """
    input:
        #net_output:{'semseg', 'semseg_logits','simmat','conf','conf_logits'}
        net_output: [ptssemseg, ptssemseg_logits, simmat_logits, conf, conf_logits]
        labels:{'ptsgroup', 'semseg','semseg_mask','group_mask'}
    """
    alpha=10.
    margin=[1.,2.]

    pts_group_label = labels['ptsgroup']
    pts_semseg_label = labels['semseg']
    group_mask = tf.expand_dims(labels['group_mask'], dim=2)

    pred_confidence_logits = net_output[3]
    pred_simmat = net_output[2]

    # Similarity Matrix loss
    B = pts_group_label.get_shape()[0]
    N = pts_group_label.get_shape()[1]

    onediag = tf.ones([B,N], tf.float32)

    group_mat_label = tf.matmul(pts_group_label,tf.transpose(pts_group_label, perm=[0, 2, 1])) #BxNxN: (i,j) if i and j in the same group
    group_mat_label = tf.matrix_set_diag(group_mat_label,onediag)

    sem_mat_label = tf.cast(tf.matmul(pts_semseg_label,tf.transpose(pts_semseg_label, perm=[0, 2, 1])), tf.float32) #BxNxN: (i,j) if i and j are the same semantic category
    sem_mat_label = tf.matrix_set_diag(sem_mat_label,onediag)

    samesem_mat_label = sem_mat_label
    diffsem_mat_label = tf.subtract(1.0, sem_mat_label)

    samegroup_mat_label = group_mat_label
    diffgroup_mat_label = tf.subtract(1.0, group_mat_label)
    diffgroup_samesem_mat_label = tf.multiply(diffgroup_mat_label, samesem_mat_label)
    diffgroup_diffsem_mat_label = tf.multiply(diffgroup_mat_label, diffsem_mat_label)

    # Double hinge loss

    C_same = tf.constant(margin[0], name="C_same") # same semantic category
    C_diff = tf.constant(margin[1], name="C_diff") # different semantic category

    pos =  tf.multiply(samegroup_mat_label, pred_simmat) # minimize distances if in the same group
    neg_samesem = alpha * tf.multiply(diffgroup_samesem_mat_label, tf.maximum(tf.subtract(C_same, pred_simmat), 0))
    neg_diffsem = tf.multiply(diffgroup_diffsem_mat_label, tf.maximum(tf.subtract(C_diff, pred_simmat), 0))


    simmat_loss = neg_samesem + neg_diffsem + pos
    group_mask_weight = tf.matmul(group_mask, tf.transpose(group_mask, perm=[0, 2, 1]))
    # simmat_loss = tf.add(simmat_loss, pos)
    simmat_loss = tf.multiply(simmat_loss, group_mask_weight)

    simmat_loss = tf.reduce_mean(simmat_loss)

    # Semantic Segmentation loss
    ptsseg_loss = tf.nn.softmax_cross_entropy_with_logits(logits=net_output[1], labels=pts_semseg_label)
    ptsseg_loss = tf.multiply(ptsseg_loss, labels['semseg_mask'])
    ptsseg_loss = tf.reduce_mean(ptsseg_loss)

    # Confidence Map loss
    Pr_obj = tf.reduce_sum(pts_semseg_label,axis=2)
    Pr_obj = tf.cast(Pr_obj, tf.float32)
    ng_label = group_mat_label
    ng_label = tf.greater(ng_label, tf.constant(0.5))
    ng = tf.less(pred_simmat, tf.constant(margin[0]))

    epsilon = tf.constant(np.ones(ng_label.get_shape()[:2]).astype(np.float32) * 1e-6)
    pts_iou = tf.div(tf.reduce_sum(tf.cast(tf.logical_and(ng,ng_label), tf.float32), axis=2),
                     (tf.reduce_sum(tf.cast(tf.logical_or(ng,ng_label), tf.float32), axis=2)+epsilon))
    confidence_label = tf.multiply(pts_iou, Pr_obj) # BxN

    confidence_loss = tf.reduce_mean(tf.squared_difference(confidence_label, tf.squeeze(pred_confidence_logits,[2])))

    loss = simmat_loss + ptsseg_loss + confidence_loss

    return loss