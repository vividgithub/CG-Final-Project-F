import tensorflow as tf
import os

_libary_path = os.path.join(__file__, "..", "ops.dll" if os.name == "nt" else "ops.so")
_op = tf.load_op_library(_libary_path)
_batch_ordered_neighbors = _op.batch_ordered_neighbors  # Legacy
_batch_grid_sub_sampling = _op.batch_grid_subsampling  # Legacy

tf.no_gradient("BatchOrderedNeighbors")
tf.no_gradient("BatchGridSubsampling")
tf.no_gradient("FixedRadiusSearch")


def _GatherDropNegatives(params, ids, zero_clipped_indices=None, is_positive=None):
  """ Helper function for unsorted segment ops.

  Gathers params for
      positive segment ids and gathers 0 for inputs with negative segment id.
      Also returns the clipped indices and a boolean mask with the same shape
      as ids where a positive id is masked as true. With this, the latter two
      can be passed as arguments to this function to reuse them.
  """
  if zero_clipped_indices is None:
    zero_clipped_indices = tf.math.maximum(ids, tf.zeros_like(ids))
  gathered = tf.gather(params, zero_clipped_indices)
  if is_positive is None:
    is_positive = tf.math.greater_equal(ids, 0)
    # tf.where(condition, x, y) requires condition to have the same shape as x
    # and y.
    # todo(philjd): remove this if tf.where supports broadcasting (#9284)
    for _ in range(gathered.shape.ndims - is_positive.shape.ndims):
      is_positive = tf.expand_dims(is_positive, -1)
    is_positive = (
        is_positive & tf.ones_like(gathered, dtype=tf.bool))
  # replace gathered params of negative indices with 0
  zero_slice = tf.zeros_like(gathered)
  return (tf.where(is_positive, gathered,
                             zero_slice), zero_clipped_indices, is_positive)


@tf.custom_gradient
def _unsorted_segment_max(data, segment_ids, num_segments, name=None):
    y = tf.math.unsorted_segment_max(data, segment_ids, num_segments, name)

    def grad_fn(grad):
        # Get the number of selected (minimum or maximum) elements in each segment.
        gathered_outputs, zero_clipped_indices, is_positive = \
            _GatherDropNegatives(y, segment_ids)
        is_selected = tf.math.equal(data, gathered_outputs)
        is_selected = tf.math.logical_and(is_selected, is_positive)
        num_selected = tf.math.unsorted_segment_sum(
            tf.cast(is_selected, grad.dtype), segment_ids, num_segments)
        # Compute the gradient for each segment. The gradient for the ith segment is
        # divided evenly among the selected elements in that segment.
        weighted_grads = tf.math.divide(grad, num_selected)
        gathered_grads, _, _ = _GatherDropNegatives(weighted_grads, None,
                                                    zero_clipped_indices, is_positive)
        zeros = tf.zeros_like(gathered_grads)
        return tf.where(is_selected, gathered_grads, zeros), None, None

    return y, grad_fn


def neighbor_aggregate(data, op, op_hint="unsorted", name=None):
    """
    Given a data (RaggedTensor) with shape (N, (neighbor), ...), reduce the "neighbor" dimension and returns
    a tensor with shape (N, ...) using the given op
    :param data: The ragged tensor for reduction
    :param op: The operation (normally unsorted_segment_min, unsorted_segment_max, ..) for reducing the op. Should
    have the same signature of "unsorted_segment_sum" or "segment_sum".
    :param op_hint: Use to tell whether we use the sorted op (like segment_max, segment_mean) or unsorted op, such as
    (unsorted_segment_max, unsorted_segment_min)
    :param name: An optional name for this op
    :return: A tensor with shape (N, ...)
    """
    with tf.name_scope(name or "NeighborAggregate"):
        segment_ids = tf.ragged.row_splits_to_segment_ids(data.row_splits)  # (Nx(neighbor), )
        if op_hint == "unsorted":
            num_segments = tf.shape(data.row_splits)[0] - 1
            return op(data.values, segment_ids, num_segments)  # (Nx(neighbor), ...) --> (N, ...)
        else:
            # Use the sorted op
            return op(data.values, segment_ids)


# Simple alias
def neighbor_aggregate_max(data, name=None):
    # The "segment_max" operation currently implements only in CPU.
    # So in order to avoid the massive memory copy form the GPU to CPU, we
    # hope to use the "unsorted_segment_max". However, in tensorflow 2.0.0,
    # there's a bug that causes the "unsorted_segment_max" cannot backpropagate
    # in eager mode. We believe this bug is caused by the wrong "OpGradientDoesntRequireOutputIndices"
    # setting for "unsorted_segment_max" in "tensorflow/python/eager/pywrap_tfe_src.cc". A
    # similar issue has been addressed in https://git.codingcafe.org/Mirrors/tensorflow/tensorflow/commit/09
    # d1dd9377a9daae3c5f14ed437d5af32b292deb?view=parallel#7dacd257e54e358c1afc8e7da12de002977dddb1.
    # We resolve this by implementing our version's "unsorted_segment_max" op.
    return neighbor_aggregate(data, op=_unsorted_segment_max,
                              op_hint="unsorted", name=name or "NeighborAggregateMax")


def neighbor_aggregate_sum(data, name=None):
    return neighbor_aggregate(data, op=tf.math.segment_sum,
                              op_hint="sorted", name=name or "NeighborAggregateSum")


def neighbor_aggregate_mean(data, name=None):
    # The segment_mean has not been implemented for GPU. So we do it by ourselves.
    with tf.name_scope(name or "NeighborAggregateMean"):
        data_sum = neighbor_aggregate_sum(data)  # (N, ...)

        row_lengths = tf.cast(data.row_splits[1:] - data.row_splits[:-1], data_sum.dtype)  # (N, )

        # Expand dim to match the data_sum
        for _ in range(data_sum.shape.ndims - row_lengths.shape.ndims):
            row_lengths = tf.expand_dims(row_lengths, -1)

        return data_sum / row_lengths


def fixed_radius_search(query_points, supported_points, query_row_splits, supported_row_splits, radius, limit, name=None):
    """
    Perform a fixed radius search. For each batch point set defined by "query_row_splits" and "supported_row_splits"
    (see RaggedTensor for more information). For that batch, we iterate each point in "query_points" and find its
    neighbor in "supported_points" where the distance is less than or equal to radius.
    :param query_points: The point to query (N, 3)
    :param supported_points: The point that to be searched (N, 3)
    :param query_row_splits: The row splits for query_points, it can be viewed as a flat RaggedTensor
    :param supported_row_splits: The row splits for supported_points
    :param radius: The radius to search
    :param limit: The point limitation, so that each point's neighbor output will be limited to that value
    :param name: An optional name for the op
    :return: A tuple (neighbor_indices, indices_row_splits), indicating a flatted RaggedTensor (N, (neighbor)), where N
    is the size of "query_points"
    """
    return _op.fixed_radius_search(query_points, supported_points, query_row_splits,
                                   supported_row_splits, radius, limit)


def grid_sampling(points, row_splits, dl, name=None):
    """
    For a given input batch points, defined by points and row_splits, sub sample each batch by constructing a
    3D grid. For each grid, we average the position of the points in the grid and output the sampled point
    :param points: An (N, 3) Tensor denoting the point set which denotes a stacked ragged batch points
    :param row_splits: The row splits for the ragged points input, a Tensor with type int and shape (B + 1, ), together
    with points that define a ragged Tensor (B, (N), 3) for input
    :param dl: The size of the grid, float
    :param name: An optional name for the op
    :return: A tuple (sampled_points, sampled_row_splits) which denotes a ragged points sub sampling output (B, (N'), 3)
    """
    # Currently it just the adapter for the legacy api

    with tf.name_scope(name or "GridSampling"):
        sampled_points, sampled_lengths = _batch_grid_sub_sampling(points, row_splits[1:] - row_splits[:-1], dl)

        sampled_row_splits = tf.math.cumsum(sampled_lengths, axis=0)
        # Prepend an 0 for the first start index for row splits
        sampled_row_splits = tf.concat([tf.constant([0], dtype=sampled_row_splits.dtype), sampled_row_splits], axis=0)

    return sampled_points, sampled_row_splits
