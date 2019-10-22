from utils.confutil import register_conf
import ops
import tensorflow as tf


@register_conf("sampling-grid", scope="layer", conf_func="self")
class GridSamplingLayer(tf.keras.layers.Layer):
    """
    A wrapper layer for grid sampling op. It accepts two input points: (N, 3) and row_splits: (B + 1, ). Similar to
    others, it defines a ragged Tensor input with shape (B + 1, (N), 3) for input. This layer generate tries use
    grid sampling strategy and generate a sub-sampled output. The output is a tuple, the first value is output_points
    with shape (N', 3) and the second is the output_row_splits with shape (B + 1). Like the input, it defines a output
    ragged Tensor with shape (B, (N'), 3), denoting the sub-sampling result
    """
    def __init__(self, dl, label=None, **kwargs):
        """
        Initialization
        :param dl: The grid size for sampling
        :param label: An optional label for the layer
        """
        super(GridSamplingLayer, self).__init__(name=label)
        self.dl = dl

    def call(self, inputs, *args, **kwargs):
        points, row_splits = inputs[0], inputs[1]
        return ops.grid_sampling(points, row_splits, dl=self.dl)