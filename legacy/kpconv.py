import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from os import makedirs
from os.path import join, exists
from legacy.utils.ply import read_ply, write_ply


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=np.sqrt(2 / shape[-1]))
    initial = tf.round(initial * tf.constant(1000, dtype=tf.float32)) / tf.constant(1000, dtype=tf.float32)
    return tf.Variable(initial, name='weights')


def kernel_point_optimization_debug(radius, num_points, num_kernels=1, dimension=3, fixed='center', ratio=1.0, verbose=0):
    """
    Creation of kernel point via optimization of potentials.
    :param radius: Radius of the kernels
    :param num_points: points composing kernels
    :param num_kernels: number of wanted kernels
    :param dimension: dimension of the space
    :param fixed: fix position of certain kernel points ('none', 'center' or 'verticals')
    :param ratio: ratio of the radius where you want the kernels points to be placed
    :param verbose: display option
    :return: points [num_kernels, num_points, dimension]
    """

    #######################
    # Parameters definition
    #######################

    # Radius used for optimization (points are rescaled afterwards)
    radius0 = 1
    diameter0 = 2

    # Factor multiplicating gradients for moving points (~learning rate)
    moving_factor = 1e-2
    continuous_moving_decay = 0.9995

    # Gradient threshold to stop optimization
    thresh = 1e-5

    # Gradient clipping value
    clip = 0.05 * radius0

    #######################
    # Kernel initialization
    #######################

    # Random kernel points
    kernel_points = np.random.rand(num_kernels * num_points - 1, dimension) * diameter0 - radius0
    while (kernel_points.shape[0] < num_kernels * num_points):
        new_points = np.random.rand(num_kernels * num_points - 1, dimension) * diameter0 - radius0
        kernel_points = np.vstack((kernel_points, new_points))
        d2 = np.sum(np.power(kernel_points, 2), axis=1)
        kernel_points = kernel_points[d2 < 0.5 * radius0 * radius0, :]
    kernel_points = kernel_points[:num_kernels * num_points, :].reshape((num_kernels, num_points, -1))

    # Optionnal fixing
    if fixed == 'center':
        kernel_points[:, 0, :] *= 0
    if fixed == 'verticals':
        kernel_points[:, :3, :] *= 0
        kernel_points[:, 1, -1] += 2 * radius0 / 3
        kernel_points[:, 2, -1] -= 2 * radius0 / 3

    #####################
    # Kernel optimization
    #####################

    # Initiate figure
    if verbose>1:
        fig = plt.figure()

    saved_gradient_norms = np.zeros((10000, num_kernels))
    old_gradient_norms = np.zeros((num_kernels, num_points))
    for iter in range(10000):

        # Compute gradients
        # *****************

        # Derivative of the sum of potentials of all points
        A = np.expand_dims(kernel_points, axis=2)
        B = np.expand_dims(kernel_points, axis=1)
        interd2 = np.sum(np.power(A - B, 2), axis=-1)
        inter_grads = (A - B) / (np.power(np.expand_dims(interd2, -1), 3/2) + 1e-6)
        inter_grads = np.sum(inter_grads, axis=1)

        # Derivative of the radius potential
        circle_grads = 10*kernel_points

        # All gradients
        gradients = inter_grads + circle_grads

        if fixed == 'verticals':
            gradients[:, 1:3, :-1] = 0

        # Stop condition
        # **************

        # Compute norm of gradients
        gradients_norms = np.sqrt(np.sum(np.power(gradients, 2), axis=-1))
        saved_gradient_norms[iter, :] = np.max(gradients_norms, axis=1)

        # Stop if all moving points are gradients fixed (low gradients diff)

        if fixed == 'center' and np.max(np.abs(old_gradient_norms[:, 1:] - gradients_norms[:, 1:])) < thresh:
            break
        elif fixed == 'verticals' and np.max(np.abs(old_gradient_norms[:, 3:] - gradients_norms[:, 3:])) < thresh:
            break
        elif np.max(np.abs(old_gradient_norms - gradients_norms)) < thresh:
            break
        old_gradient_norms = gradients_norms

        # Move points
        # ***********

        # Clip gradient to get moving dists
        moving_dists = np.minimum(moving_factor * gradients_norms, clip)

        # Fix central point
        if fixed == 'center':
            moving_dists[:, 0] = 0
        if fixed == 'verticals':
            moving_dists[:, 0] = 0

        # Move points
        kernel_points -= np.expand_dims(moving_dists, -1) * gradients / np.expand_dims(gradients_norms + 1e-6, -1)

        if verbose:
            print('iter {:5d} / max grad = {:f}'.format(iter, np.max(gradients_norms[:, 3:])))
        if verbose > 1:
            plt.clf()
            plt.plot(kernel_points[0, :, 0], kernel_points[0, :, 1], '.')
            circle = plt.Circle((0, 0), radius, color='r', fill=False)
            fig.axes[0].add_artist(circle)
            fig.axes[0].set_xlim((-radius*1.1, radius*1.1))
            fig.axes[0].set_ylim((-radius*1.1, radius*1.1))
            fig.axes[0].set_aspect('equal')
            plt.draw()
            plt.pause(0.001)
            plt.show(block=False)
            print(moving_factor)

        # moving factor decay
        moving_factor *= continuous_moving_decay

    # Rescale radius to fit the wanted ratio of radius
    r = np.sqrt(np.sum(np.power(kernel_points, 2), axis=-1))
    kernel_points *= ratio / np.mean(r[:, 1:])

    # Rescale kernels with real radius
    return kernel_points * radius, saved_gradient_norms


def load_kernels(radius, num_kpoints, num_kernels, dimension, fixed):
    # Number of tries in the optimization process, to ensure we get the most stable disposition
    num_tries = 100

    # Kernel directory
    kernel_dir = 'kernels/dispositions'
    if not exists(kernel_dir):
        makedirs(kernel_dir)

    # Kernel_file
    if dimension == 3:
        kernel_file = join(kernel_dir, 'k_{:03d}_{:s}.ply'.format(num_kpoints, fixed))
    elif dimension == 2:
        kernel_file = join(kernel_dir, 'k_{:03d}_{:s}_2D.ply'.format(num_kpoints, fixed))
    else:
        raise ValueError('Unsupported dimpension of kernel : ' + str(dimension))

    # Check if already done
    if not exists(kernel_file):

        # Create kernels
        kernel_points, grad_norms = kernel_point_optimization_debug(1.0,
                                                                    num_kpoints,
                                                                    num_kernels=num_tries,
                                                                    dimension=dimension,
                                                                    fixed=fixed,
                                                                    verbose=0)

        # Find best candidate
        best_k = np.argmin(grad_norms[-1, :])

        # Save points
        original_kernel = kernel_points[best_k, :, :]
        write_ply(kernel_file, original_kernel, ['x', 'y', 'z'])

    else:
        data = read_ply(kernel_file)
        original_kernel = np.vstack((data['x'], data['y'], data['z'])).T

    # N.B. 2D kernels are not supported yet
    if dimension == 2:
        return original_kernel

    # Random rotations depending of the fixed points
    if fixed == 'verticals':

        # Create random rotations
        thetas = np.random.rand(num_kernels) * 2 * np.pi
        c, s = np.cos(thetas), np.sin(thetas)
        R = np.zeros((num_kernels, 3, 3), dtype=np.float32)
        R[:, 0, 0] = c
        R[:, 1, 1] = c
        R[:, 2, 2] = 1
        R[:, 0, 1] = s
        R[:, 1, 0] = -s

        # Scale kernels
        original_kernel = radius * np.expand_dims(original_kernel, 0)

        # Rotate kernels
        kernels = np.matmul(original_kernel, R)

    else:

        # Create random rotations
        u = np.ones((num_kernels, 3))
        v = np.ones((num_kernels, 3))
        wrongs = np.abs(np.sum(u * v, axis=1)) > 0.99
        while np.any(wrongs):
            new_u = np.random.rand(num_kernels, 3) * 2 - 1
            new_u = new_u / np.expand_dims(np.linalg.norm(new_u, axis=1) + 1e-9, -1)
            u[wrongs, :] = new_u[wrongs, :]
            new_v = np.random.rand(num_kernels, 3) * 2 - 1
            new_v = new_v / np.expand_dims(np.linalg.norm(new_v, axis=1) + 1e-9, -1)
            v[wrongs, :] = new_v[wrongs, :]
            wrongs = np.abs(np.sum(u * v, axis=1)) > 0.99

        # Make v perpendicular to u
        v -= np.expand_dims(np.sum(u * v, axis=1), -1) * u
        v = v / np.expand_dims(np.linalg.norm(v, axis=1) + 1e-9, -1)

        # Last rotation vector
        w = np.cross(u, v)
        R = np.stack((u, v, w), axis=-1)

        # Scale kernels
        original_kernel = radius * np.expand_dims(original_kernel, 0)

        # Rotate kernels
        kernels = np.matmul(original_kernel, R)

        # Add a small noise
        kernels = kernels
        kernels = kernels + np.random.normal(scale=radius*0.01, size=kernels.shape)

    return kernels


def radius_gaussian(sq_r, sig, eps=1e-9):
    """
    Compute a radius gaussian (gaussian of distance)
    :param sq_r: input radiuses [dn, ..., d1, d0]
    :param sig: extents of gaussians [d1, d0] or [d0] or float
    :return: gaussian of sq_r [dn, ..., d1, d0]
    """
    return tf.exp(-sq_r / (2 * tf.square(sig) + eps))


def kpconv(query_points,
           support_points,
           neighbors_indices,
           features,
           K_values,
           fixed='center',
           KP_extent=1.0,
           KP_influence='linear',
           aggregation_mode='sum'):
    """
    This function initiates the kernel point disposition before building KPConv graph ops

    :param query_points: float32[n_points, dim] - input query points (center of neighborhoods)
    :param support_points: float32[n0_points, dim] - input support points (from which neighbors are taken)
    :param neighbors_indices: int32[n_points, n_neighbors] - indices of neighbors of each point
    :param features: float32[n_points, in_fdim] - input features
    :param K_values: float32[n_kpoints, in_fdim, out_fdim] - weights of the kernel
    :param fixed: string in ('none', 'center' or 'verticals') - fix position of certain kernel points
    :param KP_extent: float32 - influence radius of each kernel point
    :param KP_influence: string in ('constant', 'linear', 'gaussian') - influence function of the kernel points
    :param aggregation_mode: string in ('closest', 'sum') - whether to sum influences, or only keep the closest

    :return: output_features float32[n_points, out_fdim]
    """
    # Initial kernel extent for this layer
    K_radius = 1.5 * KP_extent

    # Number of kernel points
    num_kpoints = int(K_values.shape[0])

    # Check point dimension (currently only 3D is supported)
    points_dim = int(query_points.shape[1])

    # Create one kernel disposition (as numpy array). Choose the KP distance to center thanks to the KP extent
    K_points_numpy = load_kernels(K_radius, num_kpoints, num_kernels=1, dimension=points_dim, fixed=fixed)
    K_points_numpy = K_points_numpy.reshape((num_kpoints, points_dim))

    # Create the tensorflow variable
    K_points = tf.Variable(K_points_numpy.astype(np.float32),
                           name='kernel_points',
                           trainable=False,
                           dtype=tf.float32)

    return kpconv_ops(query_points,
                      support_points,
                      neighbors_indices,
                      features,
                      K_points,
                      K_values,
                      KP_extent,
                      KP_influence,
                      aggregation_mode)


def kpconv_ops(query_points,
               support_points,
               neighbors_indices,
               features,
               n_kp,
               K_points,
               K_values,
               KP_extent,
               KP_influence,
               aggregation_mode):
    """
    This function creates a graph of operations to define Kernel Point Convolution in tensorflow. See KPConv function
    above for a description of each parameter

    :param query_points:        [n_points, dim]
    :param support_points:      [n0_points, dim]
    :param neighbors_indices:   [n_points, n_neighbors]
    :param features:            [n0_points, in_fdim]
    :param n_kp:                int
    :param K_points:            [n_kpoints, dim]
    :param K_values:            [n_kpoints, in_fdim, out_fdim]
    :param KP_extent:           float32
    :param KP_influence:        string
    :param aggregation_mode:    string
    :return:                    [n_points, out_fdim]
    """

    # Get variables
    n_kp = int(K_points.shape[0])

    # Add a fake point in the last row for shadow neighbors
    shadow_point = tf.ones_like(support_points[:1, :]) * 1e6
    support_points = tf.concat([support_points, shadow_point], axis=0)

    # Get neighbor points [n_points, n_neighbors, dim]
    neighbors = tf.gather(support_points, neighbors_indices, axis=0)

    # Center every neighborhood
    neighbors = neighbors - tf.expand_dims(query_points, 1)

    # Get all difference matrices [n_points, n_neighbors, n_kpoints, dim]
    neighbors = tf.expand_dims(neighbors, 2)
    neighbors = tf.tile(neighbors, [1, 1, n_kp, 1])
    differences = neighbors - K_points

    # Get the square distances [n_points, n_neighbors, n_kpoints]
    sq_distances = tf.reduce_sum(tf.square(differences), axis=3)

    # Get Kernel point influences [n_points, n_kpoints, n_neighbors]
    if KP_influence == 'constant':
        # Every point get an influence of 1.
        all_weights = tf.ones_like(sq_distances)
        all_weights = tf.transpose(all_weights, [0, 2, 1])

    elif KP_influence == 'linear':
        # Influence decrease linearly with the distance, and get to zero when d = KP_extent.
        all_weights = tf.maximum(1 - tf.sqrt(sq_distances) / KP_extent, 0.0)
        all_weights = tf.transpose(all_weights, [0, 2, 1])

    elif KP_influence == 'gaussian':
        # Influence in gaussian of the distance.
        sigma = KP_extent * 0.3
        all_weights = radius_gaussian(sq_distances, sigma)
        all_weights = tf.transpose(all_weights, [0, 2, 1])
    else:
        raise ValueError('Unknown influence function type (config.KP_influence)')

    # In case of closest mode, only the closest KP can influence each point
    if aggregation_mode == 'closest':
        neighbors_1nn = tf.argmin(sq_distances, axis=2, output_type=tf.int32)
        all_weights *= tf.one_hot(neighbors_1nn, n_kp, axis=1, dtype=tf.float32)

    elif aggregation_mode != 'sum':
        raise ValueError("Unknown convolution mode. Should be 'closest' or 'sum'")

    features = tf.concat([features, tf.zeros_like(features[:1, :])], axis=0)

    # Get the features of each neighborhood [n_points, n_neighbors, in_fdim]
    neighborhood_features = tf.gather(features, neighbors_indices, axis=0)

    # Apply distance weights [n_points, n_kpoints, in_fdim]
    weighted_features = tf.matmul(all_weights, neighborhood_features)

    # Apply network weights [n_kpoints, n_points, out_fdim]
    weighted_features = tf.transpose(weighted_features, [1, 0, 2])
    kernel_outputs = tf.matmul(weighted_features, K_values)

    # Convolution sum to get [n_points, out_fdim]
    output_features = tf.reduce_sum(kernel_outputs, axis=0)

    return output_features