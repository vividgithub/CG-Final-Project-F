import tensorflow as tf
import logger
from legacy.kpconv import kpconv_ops, load_kernels
from utils.confutil import register_conf


@register_conf(name="conv-kp", scope="layer", conf_func="self")
class KPConvLayer(tf.keras.layers.Layer):
    """The KP convolution layer(https://arxiv.org/pdf/1904.08889.pdf)"""

    def __init__(self, channel, k, extent, fixed="center", influence="linear", aggregation="sum", **kwargs):
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
        # points: (N, 3)
        # features: (N, F)
        # ouput_points: (N', 3)
        # neighbor_indices: (N', (neighbor)), should be global indices
        # output: (N', channel)
        points, features, output_points, neighbor_indices = inputs

        neighbors = tf.gather(tf.concat([points, features], axis=-1), neighbor_indices)  # (N', (neighbor), F + 3)

        # Get all difference matrices
        neighbors = tf.expand_dims(neighbors, axis=2)  # (N', (neighbor), 1, F + 3)
        neighbors = tf.tile(neighbors, [1, 1, self.k, 1])  # (N', (neighbor), k, F + 3)

        # Unpack the position and features
        # neighbors: (N', (neighbor), k, 3)
        # neighbor_features: (N', (neighbor), k, F)
        neighbors, neighbor_features = neighbors[..., :3], neighbors[..., 3:]

        # differences: (N', (neighbor), k, 3)
        differences = neighbors - output_points[..., tf.newaxis, tf.newaxis, :]  # Convert global to relative positions
        differences = differences - self.k_points  # Convert the positions relative to the kernel points

        # Get the square distances
        sq_distances = tf.reduce_sum(tf.square(differences), axis=-1)  # (N', (neighbor), k)

        # Get Kernel point influences
        assert self.influence == "linear", \
            f"KP convolution only support linear influence, get \"{self.influence}\""
        all_weights = tf.maximum(1 - tf.sqrt(sq_distances) / self.extent, 0.0)  # (N', (neighbor), k)
        all_weights = tf.expand_dims(all_weights, axis=-1)  # (N', (neighbor), k, 1)

        weighted_features = all_weights * neighbor_features  # (N', (neighbor), k, F)
        weighted_features = tf.reduce_sum(weighted_features, axis=1)  # (N', k, F)

        # Apply network weights
        weighted_features = tf.transpose(weighted_features, [1, 0, 2])  # (k, N', F)
        kernel_outputs = tf.matmul(weighted_features, self.k_values)  # (k, N', F) x (k, F, channel) -> (k, N', channel)

        output_features = tf.reduce_sum(kernel_outputs, axis=0)
        return output_features
