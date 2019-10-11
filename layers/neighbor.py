import ops
from utils.confutil import register_conf
import tensorflow as tf


@register_conf("neighbor-fixed", scope="layer", conf_func="self")
class FixedRadiusNeighborQuery(tf.keras.layers.Layer):
    """
    A wrapper layer for ops fixed radius search. It accepts four parameter points: (N, 3), row_splits: (B + 1, ),
    output_points: (N', 3) and output_row_splits: (B + 1, ). Those four inputs are actually two ragged Tensor input,
    which is (B, (N), 3) and (B, (N'), 3). The output_points defines the position that need to perform the radius
    search, and the points defines actual point set. Such design allows us to query neighbor information for a point
    set itself (by making output_points = points and output_row_splits = row_splits). And also it supports to query
    neighbor information only for a subset for the point set (by making output_points and output_row_splits) equals
    to the sub sampling set for the original point set.
    """
    def __init__(self, radius, limit, label=None, **kwargs):
        """
        Initialization
        :param radius: The search radius for the neighbor
        :param limit: The maximum neighbor output for each point, it is used to limit the total output point if
        some area is really dense
        :param label: An optional label for the layer
        """
        super(FixedRadiusNeighborQuery, self).__init__(name=label)
        self.radius = radius
        self.limit = limit

    def call(self, inputs, *args, **kwargs):
        points, row_splits = inputs[0], inputs[1]
        output_points, output_row_splits = inputs[2], inputs[3]

        # Note that the ops use a different input order
        return ops.fixed_radius_search(output_points, points, output_row_splits,
                                       row_splits, radius=self.radius, limit=self.limit)