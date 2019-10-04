import tensorflow as tf
import logger
from legacy.kpconv import kpconv_ops, load_kernels


class KPConvLayer(tf.keras.layers.Layer):
    """The KP convolution layer(https://arxiv.org/pdf/1904.08889.pdf)"""

    def __init__(self, channel, k, extent, fixed="center", influence="linear", aggregation="sum"):
        """
        Initialize a KP convolution layer
        :param channel: Number of channel to output
        :param k: Number of kernel points in the convolution
        :param extent: The influence radius of the kernel points
        :param fixed: String in ('none', 'center' or 'verticals') - fix position of certain kernel points
        :param influence: String in ('constant', 'linear', 'gaussian') - influence function of the kernel points
        :param aggregation: String in ('closest', 'sum') - whether to sum influences, or only keep the closest
        """
        self.channel = channel
        self.k = k
        self.extent = extent
        self.fixed = fixed
        self.influence = influence
        self.aggregation = aggregation
        self.k_values = None

        # Load the k_points location
        self.k_points = load_kernels(1.5 * extent, k, num_kernels=1, dimension=3, fixed=fixed).reshape(k, 3)
        logger.log("Kernel point, extent={}, size={}, values={}".format(extent, k, self.k_points))
        self.k_points = tf.constant(self.k_points, dtype=tf.float32)

    def build(self, input_shapes):
        # Inputs should be
        #   points (B, N, 3),
        #   features (B, N, F),
        #   output_points (B, N', 3)
        #   neighbor_indices (B, N', #neighbor))
        #
        # Add a weight with size (k, F, channel) to the output
        f = input_shapes[1][-1]
        self.k_values = self.add_weight(
            name="k_values",
            shape=(self.k, f, self.channel),
            trainable=True
        )

    def call(self, inputs, *args, **kwargs):
        # points: (B, N, 3)
        # features: (B, N, F)
        # ouput_points: (B, N', 3)
        # neighbor_indices: (B, N', #neighbor)
        # output: (B, N', channel)
        points, features, output_points, neighbor_indices = inputs

        b = tf.shape(points)[0]

        # Squeeze the batch
        neighbors = tf.gather(points, neighbor_indices, batch_dims=1)  # (B, N', #neighbor, 3)
        neighbors = neighbors - tf.expand_dims(output_points, axis=2)  # (B, N', #neighbor, 3), centering

        # Get all difference matrices
        neighbors = tf.expand_dims(neighbors, axis=3)  # (B, N', #neighbor, 1, 3)
        neighbors = tf.tile(neighbors, [1, 1, 1, self.k, 1])  # (B, N', #neighbor, k, 3)
        differences = neighbors - self.k_points  # (B, N', #neighbor, k, 3)

        # Get the square distances
        sq_distances = tf.reduce_sum(tf.square(differences), axis=-1)  # (B, N', #neighbor, k)

        # Get Kernel point influences
        assert self.influence == "linear", \
            "KP convolution only support linear influence, get \"{}\"".format(self.influence)
        all_weights = tf.maximum(1 - tf.sqrt(sq_distances) / self.extent, 0.0)  # (B, N', #neighbor, k)
        all_weights = tf.transpose(all_weights, [0, 1, 3, 2])  # (B, N', k, #neighbor)

        # Aggregation
        neighbor_features = tf.gather(features, neighbor_indices, batch_dims=1)  # (B, N', #neighbor, F)
        weighted_features = tf.matmul(all_weights, neighbor_features)  # (B, N', k, F)

        # Apply network weights
        weighted_features = tf.transpose(weighted_features, [1, 2, 0, 3])  # (k, B, N', F)
        k_values = self.k_values  # (k, F, channel)
        k_values = tf.expand_dims(k_values, axis=1)  # (k, 1, F, channel)
        k_values = tf.tile(k_values, [1, b, 1, 1])  # (k, B, F, channel)
        kernel_outputs = tf.matmul(weighted_features, k_values)  # (k, B, N', F) x (k, B, F, channel) -> (k, B, N', channel)

        output_features = tf.reduce_sum(kernel_outputs, axis=0)
        return output_features

