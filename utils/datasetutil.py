import tensorflow as tf
from os import path
import h5py
from logger import log
from utils import ioutil
from utils.confutil import register_conf, object_from_conf
import numpy as np


def apply_transforms(dataset, confs, batch_size):
    """
    Apply transforms to the dataset.
    :param confs: A list of configuration like {"name": "scaling", ... }, indicating the specified transform. We use
    "--------" to separate the pre-batch-transform and after-batch-transform.
    :param batch_size: The batch size to batch the dataset
    :return: A new dataset after transform
    """
    # Per point cloud instance transform
    pre_batch_confs = confs
    post_batch_confs = []
    try:
        batch_seperator_index = [type(x) is str and "---" in x for x in confs].index(True)
        pre_batch_confs = confs[:batch_seperator_index]
        post_batch_confs = confs[batch_seperator_index + 1:]
    except ValueError:
        # Cannot find the index
        pass

    log("Origin dataset={}".format(dataset))
    context = {"batch_size": batch_size}
    for conf in pre_batch_confs:
        transform = object_from_conf(conf, scope="transform", context=context)
        dataset = transform(dataset)
        log("After pre-batch transform \"{}\" with conf={}, dataset={}".format(conf["name"], conf, dataset))

    dataset = dataset.batch(batch_size)
    log("Batch transform, dataset={}".format(dataset))

    for conf in post_batch_confs:
        transform = object_from_conf(conf, scope="transform", context=context)
        dataset = transform(dataset)
        log("After post-batch transform \"{}\" with conf={}, dataset={}".format(conf["name"], conf, dataset))

    return dataset


def load_dataset_h5(dir, data_conf, train_load_policy="normal", test_load_policy="normal", **kwargs):
    """
    Load a h5 dataset from specified folder "dir", the h5 dataset should contain a "train.h5" (or other name specified
    by conf) and a "test.h5" (or other name specified by conf).

    :param dir: The directory of the dataset
    :param data_conf: The configuration dict of dataset
    :param train_load_policy: The load policy that uses for loading the dataset. "normal" tries to load the dataset
    in the ordinary way. "random" tries to randomly sample from the dataset. "random-balance" tries to randomly
    sample through classes. That is, we try to sample each class with the same possibility, regardless of its
    number of data.
    :return: Two dataset (train_dataset, test_dataset). Each dataset will yield (points, label) for each iterations,
    where "points" is a (N,F) tensor(N: number of points, F: number of features, including positions) and "label" is a
    (N,) vector for segmentation task and a (1,) for classification task. When the loading policy is "normal",
    it will return a finite dataset. When the loading policy set to "random" or "random-balance", it will return an
    infinite random dataset.
    """
    # TODO: random policy
    assert train_load_policy == "normal" and test_load_policy == "normal", "We don't support random policy right now"
    assert test_load_policy == "normal", "Test policy should be normal otherwise it will return an infinite dataset"

    train_filename = data_conf.get("type", dict()).get("train_file") or "train.h5"
    test_filename = data_conf.get("type", dict()).get("test_file") or "test.h5"

    train_filepath = path.join(dir, train_filename)
    test_filepath = path.join(dir, test_filename)
    log("Loading train file=\"{}\", policy={}".format(train_filepath, train_load_policy))
    log("Loading test file \"{}\", policy={}".format(test_filepath, test_load_policy))

    def _h5_generator(filepath, policy):
        f = h5py.File(filepath)
        if policy == "normal":
            for points, label, seglabel in zip(f["data"],  f["pid"], f["seglabel"] ):

                yield points, label, seglabel# shape(4096, 9),(4096, ),(4096, )
        else:
            assert False, "random policy is not supported"
        f.close()

    # def _train_h5_generator(filepath, policy):
    #     f = h5py.File(filepath)
    #     for points, label, seglabel in zip(f["data"], f["pid"], f["seglabel"]):
    #         yield points, label, seglabel  # shape(4096, 9),(4096, ),(4096, )

    #     f.close()

    train_gen = lambda: _h5_generator(train_filepath, train_load_policy)
    test_gen = lambda: _h5_generator(test_filepath, test_load_policy)
    output_types = (tf.float32, tf.int64)

    point_count = data_conf.get("point_count", None)
    feature_size = data_conf.get("feature_size", None)
    batch_size = 80
    # output_shapes = ([batch_size, point_count, feature_size], [batch_size, point_count], [batch_size, point_count])
    # output_shapes = (batch_size, point_count, feature_size + 2)

    f = h5py.File(train_filepath)
    points, label, seglabel = f["data"],  f["pid"], f["seglabel"]

    train_dataset = tf.data.Dataset.from_tensor_slices((points, np.append(np.array(label)[:,np.newaxis,:], np.array(seglabel)[:,np.newaxis,:], axis = 1)))
    f.close()

    f = h5py.File(test_filepath)
    points, label, seglabel = f["data"],  f["pid"], f["seglabel"]
    test_dataset = tf.data.Dataset.from_tensor_slices((points, np.append(np.array(label)[:,np.newaxis,:], np.array(seglabel)[:,np.newaxis,:], axis = 1)))
    f.close()

    return (
        train_dataset,
        test_dataset
    )


def load_dataset(dir, model_conf):
    """
    Load a dataset from a specified directory
    :param dir: The directory of the dataset, make sure that it has a "conf.pyconf" file
    :param model_conf: The configuration dictionary of model
    :return: A tuple (train_dataset, test_dataset, conf) where conf is the data configuration
    """
    loaders = {
        "dataset-h5": load_dataset_h5
    }

    dir = path.abspath(dir)

    conf_path = path.join(dir, "conf.pyconf")
    conf = ioutil.pyconf(conf_path)

    assert conf["type"]["name"] in loaders, "Invalid dataset type for {}".format(conf["type"])

    train_dataset, test_dataset = loaders[conf["type"]["name"]](dir, conf, **model_conf)
    batch_size = model_conf["control"]["batch_size"]

    # train_dataset = pass
    # train_dataset = apply_transforms(train_dataset, model_conf["dataset"].get("train_transforms", []), batch_size)
    # test_dataset = apply_transforms(test_dataset, model_conf["dataset"].get("test_transforms", []), batch_size)
    print(train_dataset)
    print(test_dataset)
    exit()
    return train_dataset, test_dataset, conf




def dataset_transform(func):
    """
    This decorator accepts a dataset transform function "lambda dataset: ..." and print info
    :param func: The dataset transform function, which should accepts an dataset and output a dataset
    :return: func itself
    """
    def _func(*args, **kwargs):
        log(
            "Generating dataset map transform \"{}({}{})\"".format(
                func.__name__,
                ",".join([str(arg) for arg in args]) + (", " if len(args) > 0 else ""),
                ",".join([str(key) + "=" + str(value) for key, value in kwargs.items()])
            )
        )
        return func(*args, **kwargs)
    return _func


def dataset_map_transform(func):
    """
    This decorator converts a transform function to a dataset map callable lambda and print info
    :param func: The map function that needs to decorate
    :return: A mapper, which mapper(dataset) = dataset.map(some_function)
    """
    def _func(*args, **kwargs):
        log(
            "Generating dataset map transform \"{}({}{})\"".format(
                func.__name__,
                ",".join([str(arg) for arg in args]) + (", " if len(args) > 0 else ""),
                ",".join([str(key) + "=" + str(value) for key, value in kwargs.items()])
            )
        )
        return lambda dataset: dataset.map(func(*args, **kwargs))
    return _func


@register_conf(name="clip-feature", scope="transform", conf_func="self")
@dataset_map_transform
def transform_clip_feature(c=3, **kwargs):
    """
    Given an input with NxF point features, extract the first "c" features and discard the left
    :param c: The number of features to maintain
    :return: A clip feature function (NxF, _) -> (Nxc, _)
    """
    return lambda points, label: (points[:, :c, ...], label)


@register_conf(name="sampling", scope="transform", conf_func="self")
@dataset_map_transform
def transform_sampling(sample_num, policy="random-gauss", range=None, stddev=None, **kwargs):
    """
    Given an input with BxNxF point features, random sampling the point cloud
    :param sample_num: The base sample num, when "policy" is normal, it is the number of points that we should return.
    For "random" sampling, this should be the mean value.
    :param policy: The sampling policy, "normal" will extract the first [0,sample_num) points, while
    "random-gauss" will random sample on the point cloud using normal distribution, while "random-uniform"
    will randomly sample on point cloud using uniform distribution
    :param range: Use in random sampling, a tuple (min_sample_num, max_sample_num) to specify the sample range
    :param stddev" Use in "random-gauss" policy, the standard deviation for normal distribution
    :return: A sampling function (BxNxF, _) -> (BxN'xF, _), where N' is the actual sampling number
    """

    # def _transform_sampling(points, sample_num, policy, variance, clip):
    #     offset = int(random.gauss(0, sample_num * variance))
    #     offset = max(offset, int(-sample_num * clip))
    #     offset = min(offset, int(sample_num * clip))
    #
    #     n_ = sample_num + offset
    #     n = tf.shape(points)[1]
    #     assert n > n_, "The sampled number n_={} > n={}".format(n_, n)
    #
    #     sampling_func = lambda: np.arange(n_) if policy == "normal" \
    #         else np.random.choice(n, n_, replace=False)
    #
    #     sample_points = np.stack([p[sampling_func(), ...] for p in points.numpy()])
    #     return sample_points
    #
    # # Get a curried _transform_sampling to use it in tf.py_function
    # _transform_sampling_closure = lambda points, label: (_transform_sampling(points, sample_num, policy, variance, clip), label)
    # return lambda points, label: tf.py_function(_transform_sampling_closure, inp=[points, label], Tout=(tf.float32, tf.int64))

    # def _random_sample(points, n, n_):
    #     sampling_func = lambda: np.random.choice(n, n_, replace=False)
    #     sample_points = np.stack([p[sampling_func(), ...] for p in points.numpy()])
    #     return sample_points
    # random_sample = lambda points, n, n_: tf.py_function(_random_sample, inp=(points, n, n_), Tout=tf.float32)

    def _transform_sampling_normal(points, label):
        return points[:, :sample_num, :], label

    def _transform_sampling_random_gauss(points, label):
        shape = tf.shape(points)
        b = shape[0]
        n = shape[1]
        n_ = tf.cast(
            tf.clip_by_value(
                tf.random.normal((), mean=sample_num, stddev=stddev, dtype=tf.float32),
                range[0],
                range[1]
            ),
            dtype=tf.int32
        )

        # (B, N, F) -> (B, N, index)
        # This method will use the same indices for all the point cloud in the same batch, it won't cause
        # much trouble but it should make sure that the original dataset have shuffled enough
        # Currently, we allow a point to be sampled multiple times
        samples = tf.random.uniform((b, n_), maxval=n, dtype=tf.int32)  # (B, N')

        batch_indices = tf.range(b, dtype=tf.int32)  # (B, ): [0, 1, 2, ..., b-1]
        batch_indices = batch_indices[:, tf.newaxis]  # (B, 1): [[0], [1], [2], ..., [b-1]]
        batch_indices = tf.tile(batch_indices, (1, n_))  # (B, N_): [[0, 0, ...], [1, 1, ...], ..., [b-1, b-1, ...]]

        indices = tf.stack([batch_indices, samples], axis=-1)  # (B, N_) ~stack~ (B, N_) = (B, N_, 2)
        points_ = tf.gather_nd(points, indices)  # (B, N, F) ~gather_nd~ (B, N_, 2) -> (B, N_, F)
        return points_, label

    if policy == "random-gauss":
        assert range is not None and stddev is not None, "Policy \"random-gauss\" should specify the \"range\" and \"stddev\""
        return _transform_sampling_random_gauss
    elif policy == "normal":
        return _transform_sampling_normal
    else:
        assert False, "Policy \"{}\" is unsupported"


@register_conf(name="scaling", scope="transform", conf_func="self")
@dataset_map_transform
def transform_scaling(range=(0.0, 0.05), anisotropic=True, **kwargs):
    """
    Given an input of NxF, random scaling the coordinate. That is, transform the first 3 features but remains the
    other features unchanged
    :param range: The random scaling range
    :param anisotropic: Whether to use different scale factor for each axis
    :return: A function (NxF, _) -> (NxF, _), where the return point features have been transformed.
    "_" indicates the label input will remain unchanged.
    """

    return lambda points, label: (
        tf.concat(
            [
                points[..., :3] * tf.random.uniform((3, ) if anisotropic else (), minval=range[0], maxval=range[1]),
                points[..., 3:]
            ],
            axis=-1
        ),
        label
    )


@register_conf(name="rotation", scope="transform", conf_func="self")
@dataset_map_transform
def transform_rotation(policy, range, **kwargs):
    """
    Given an input of NxF, random rotate the coordinate. That is, transform the first 3 features but maintain the others.
    :param policy: The rotation policy, "euler" uses "Euler angle rotation", and "quaternion" uses "quaternion" for
    rotation.
    :param range: The rotation range. For "euler" policy, it should be a list [(xmin,xmax), (ymin, ymax), (zmin,zmax)],
    the angle for rotation in each axis (in radians). For "quaternion" policy, ...
    :return: A function (NxF, _) -> (NxF, _), where the return point features have been transformed.
    "_" indicates the label input will remain unchanged.
    """
    def _transform_rotation_euler(points, label):
        rx = tf.random.uniform((), minval=range[0][0], maxval=range[0][1])
        ry = tf.random.uniform((), minval=range[1][0], maxval=range[1][1])
        rz = tf.random.uniform((), minval=range[2][0], maxval=range[2][1])

        cx = tf.math.cos(rx)
        cy = tf.math.cos(ry)
        cz = tf.math.cos(rz)

        sx = tf.math.sin(rx)
        sy = tf.math.sin(ry)
        sz = tf.math.sin(rz)

        matx = tf.stack([
            [1, 0, 0],
            [0, cx, -sx],
            [0, sx, cx]
        ])
        maty = tf.stack([
            [cy, 0, sy],
            [0, 1, 0],
            [-sy, 0, cy]
        ])
        matz = tf.stack([
            [cz, -sz, 0],
            [sz, cz, 0],
            [0, 0, 1]
        ])
        matx = matx[tf.newaxis, ...]  # (1, 3, 3)
        maty = maty[tf.newaxis, ...]  # (1, 3, 3)
        matz = matz[tf.newaxis, ...]  # (1, 3, 3)
        mat = tf.linalg.matmul(tf.linalg.matmul(matz, maty), matx)  # (1, 3, 3)
        mat = tf.tile(mat, (tf.shape(points)[0], 1, 1))  # (N, 3, 3)
        pos = points[:, :3, tf.newaxis]  # (N, 3, 1)

        pos_ = tf.linalg.matmul(mat, pos)  # (N, 3, 3) x (N, 3, 1) -> (N, 3, 1)
        return tf.concat([tf.squeeze(pos_, axis=-1), points[:, 3:]], axis=-1), label  # (N, 3) ~concat~ (N, F - 3) -> (N, F)

    if policy == "euler":
        return _transform_rotation_euler


@register_conf(name="shuffle", scope="transform", conf_func="self")
@dataset_transform
def transform_shuffle(batch_size, buffer_multiplier=4, **kwargs):
    """
    A wrapper for dataset shuffle
    :param batch_size: The provided batch size
    :param buffer_multiplier: A multiplier to the batch size.
    The shuffle the buffer size is equal to "batch_size * buffer_multiplier"
    :return: A transform function
    """

    return lambda dataset: dataset.shuffle(buffer_size=batch_size * buffer_multiplier)


@register_conf(name="repeat", scope="transform", conf_func="self")
@dataset_transform
def transform_repeat(**kwargs):
    """
    A wrapper for dataset repeat
    :return: A transform function
    """
    return lambda dataset: dataset.repeat()
