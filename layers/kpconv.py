import tensorflow as tf
import logger
from legacy.kpconv import kpconv_ops, load_kernels
from utils.confutil import register_conf


@register_conf(name="conv-kp", scope="layer", conf_func="self")
class KPConvLayer(tf.keras.layers.Layer):
    """The KP convolution layer(https://arxiv.org/pdf/1904.08889.pdf)"""

    def __init__(self, channel, k, extent, fixed="center", influence="linear", aggregation="sum", label=None, **kwargs):
        """
        Initialize a KP convolution layer
        :param channel: Number of channel to output
        :param k: Number of kernel points in the convolution
        :param extent: The influence radius of the kernel points
        :param fixed: String in ('none', 'center' or 'verticals') - fix position of certain kernel points
        :param influence: String in ('constant', 'linear', 'gaussian') - influence function of the kernel points
        :param aggregation: String in ('closest', 'sum') - whether to sum influences, or only keep the closest
        :param label: An optional label for the layer
        """
        super(KPConvLayer, self).__init__(name=label)
        self.channel = channel
        self.k = k
        self.extent = extent
        self.fixed = fixed
        self.influence = influence
        self.aggregation = aggregation
        self.k_values = None

        # Load the k_points location
        self.k_points = load_kernels(1.5 * extent, k, num_kernels=1, dimension=3, fixed=fixed).reshape(k, 3)
        self.k_points = tf.constant(self.k_points, dtype=tf.float32)

    def build(self, input_shapes):
        # Inputs should be
        #   points (N, 3),
        #   features (N, F),
        #   output_points (N', 3)
        #   neighbor_indices (N', (neighbor))
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

        # Split the positions and features
        neighbors = tf.gather(points, neighbor_indices)  # (N', (neighbor), 3)
        neighbors_features = tf.gather(features, neighbor_indices)  # (N', (neighbor), F)

        # Convert to local coordinates
        neighbors = neighbors - tf.expand_dims(output_points, axis=1)  # (N', (neighbor), 3)

        # Flatten
        neighbors_row_splits = neighbors.row_splits
        neighbors, neighbors_features = neighbors.values, neighbors_features.values

        # Get all difference matrices
        neighbors = tf.expand_dims(neighbors, axis=1)  # (N'x(neighbor), 1, 3)
        neighbors = tf.tile(neighbors, [1, self.k, 1])  # (N'x(neighbor), k, 3)
        neighbors_features = tf.expand_dims(neighbors_features, axis=1)  # (N'x(neighbor), 1, F)
        neighbors_features = tf.tile(neighbors_features, [1, self.k, 1])  # (N'x(neighbor), k, F)

        # differences: (N'x(neighbor), k, 3)
        differences = neighbors - self.k_points  # Convert the positions relative to the kernel points

        # Get the square distances
        sq_distances = tf.reduce_sum(tf.square(differences), axis=-1)  # (N'x(neighbor), k)

        # Get Kernel point influences
        assert self.influence == "linear", \
            f"KP convolution only support linear influence, get \"{self.influence}\""
        all_weights = tf.maximum(1 - tf.sqrt(sq_distances) / self.extent, 0.0)  # (N'x(neighbor), k)
        all_weights = tf.expand_dims(all_weights, axis=-1)  # (N'x(neighbor), k, 1)

        weighted_features = all_weights * neighbors_features  # (N'x(neighbor), k, F)
        weighted_features = tf.reduce_sum(
            tf.RaggedTensor.from_row_splits(weighted_features, neighbors_row_splits),
            axis=1
        )  # (N'x(neighbor), k, F) --> (N', (neighbor), k, F) --> (N', k, F)

        # Apply network weights
        weighted_features = tf.transpose(weighted_features, [1, 0, 2])  # (k, N', F)
        kernel_outputs = tf.matmul(weighted_features, self.k_values)  # (k, N', F) x (k, F, channel) -> (k, N', channel)

        output_features = tf.reduce_sum(kernel_outputs, axis=0)
        return output_features
