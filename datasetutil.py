import tensorflow as tf
from os import path, listdir
import h5py
import numpy as np
from logger import log
import random
import legacy.pointfly as pf
import transforms3d
import ioutil


def apply_transforms(dataset, confs, batch_size):
    """
    Apply transforms to the dataset.
    :param confs: A list of configuration like {"name": "scaling", ... }, indicating the specified transform. We use
    "--------" to separate the pre-batch-transform and after-batch-transform.
    :param batch_size: The batch size to batch the dataset
    :return: A new dataset after transform
    """
    transform_map = {
        "clip-feature": transform_clip_feature,
        "sampling": transform_sampling,
        "scaling": transform_scaling,
        "rotation": transform_rotation
    }

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

    for conf in pre_batch_confs:
        assert transform_map.get(conf["name"]), "Unable to find the transform \"{}\"".format(conf["name"])
        map_func = transform_map[conf["name"]](**conf)
        dataset = dataset.map(map_func)

    dataset = dataset.batch(batch_size)

    for conf in post_batch_confs:
        assert transform_map.get(conf["name"]), "Unable to find the transform \"{}\"".format(conf["name"])
        map_func = transform_map[conf["name"]](**conf)
        dataset = dataset.map(map_func)

    return dataset


def load_dataset_h5(dir, conf, train_load_policy="normal", test_load_policy="normal", **kwargs):
    """
    Load a h5 dataset from specified folder "dir", the h5 dataset should contain a "train.h5" (or other name specified
    by conf) and a "test.h5" (or other name specified by conf).

    :param dir: The directory of the dataset
    :param conf: The configuration dict
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

    train_filename = conf.get("type", dict()).get("train_file") or "train.h5"
    test_filename = conf.get("type", dict()).get("test_file") or "test.h5"

    train_filepath = path.join(dir, train_filename)
    test_filepath = path.join(dir, test_filename)
    log("Loading train file=\"{}\", policy={}".format(train_filepath, train_load_policy))
    log("Loading test file \"{}\", policy={}".format(test_filepath, test_load_policy))

    def _h5_generator(filepath, policy):
        f = h5py.File(filepath)
        if policy == "normal":
            for points, label in zip(f["data"], f["label"]):
                yield points, label
        else:
            assert False, "random policy is not supported"
        f.close()

    train_gen = lambda: _h5_generator(train_filepath, train_load_policy)
    test_gen = lambda: _h5_generator(test_filepath, test_load_policy)
    output_types = (tf.float32, tf.int64)

    return (
        tf.data.Dataset.from_generator(train_gen, output_types=output_types),
        tf.data.Dataset.from_generator(test_gen, output_types=output_types)
    )


def load_dataset(dir, model_conf):
    """
    Load a dataset from a specified directory
    :param dir: The directory of the dataset, make sure that it has a "conf.pyconf" file
    :param model_conf: The configuration dictionary of model
    :return: A tuple (dataset, dict) where dict is the configuration
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

    train_dataset = apply_transforms(train_dataset, model_conf["dataset"].get("train_transforms", []), batch_size)
    test_dataset = apply_transforms(test_dataset, model_conf["dataset"].get("test_transforms", []), batch_size)

    return train_dataset, test_dataset, conf


def transform_clip_feature(c=3, **kwargs):
    """
    Given an input with NxF point features, extract the first "c" features and discard the left
    :param c: The number of features to maintain
    :return: A clip feature function (NxF) -> (Nxc)
    """
    return lambda points, label: (points[:, :c, ...], label)


def dataset_transform(func):
    """
    The dataset transform function decorator that prints transform functino info
    :param func: The function that needs to decorate
    :return: The decorated function
    """
    def _func(*args, **kwargs):
        log(
            "Generating dataset transform \"{}({}{})\"".format(
                func.__name__,
                ",".join([str(arg) for arg in args]) + (", " if len(args) > 0 else ""),
                ",".join([str(key) + "=" + str(value) for key, value in kwargs.items()])
            )
        )
        return func(*args, **kwargs)
    return _func


@dataset_transform
def transform_sampling(sample_num, policy="random", variance=0.0, clip=0.0, **kwargs):
    """
    Given an input with BxNxF point features, random sampling the point cloud
    :param sample_num: The base sample num.
    :param policy: The sampling policy, "normal" will extract the first [0,sample_num) points, while
    "random" will random sample on the point cloud
    :param variance: The standard deviation of the guass sampling kernel
    :param clip: The clip factor, the final point will not less that sample_num * (1 - clip) and not more than
    sample_num * (1 + clip)
    :return: A sampling function (BxNxF, _) -> (BxN'xF, _), where N' is the actual sampling number,
    N' is a random variable of normal distribution with mean = sample_num and stddev = variance. N' will be
    further clipped to ensure that it's value is not larger than sample_num * (1 + clip) and not smaller than
    sample_num * (1 - clip). "_" indicates the label input will remain unchanged.
    """

    def _transform_sampling(points, sample_num, policy, variance, clip):
        offset = int(random.gauss(0, sample_num * variance))
        offset = max(offset, int(-sample_num * clip))
        offset = min(offset, int(sample_num * clip))

        n_ = sample_num + offset
        n = tf.shape(points)[1]
        assert n > n_, "The sampled number n_={} > n={}".format(n_, n)

        sampling_func = lambda: np.arange(n_) if policy == "normal" \
            else np.random.choice(n, n_, replace=False)

        sample_points = np.stack([p[sampling_func(), ...] for p in points.numpy()])
        return sample_points

    # Get a curried _transform_sampling to use it in tf.py_function
    _transform_sampling_closure = lambda points, label: (_transform_sampling(points, sample_num, policy, variance, clip), label)
    return lambda points, label: tf.py_function(_transform_sampling_closure, inp=[points, label], Tout=(tf.float32, tf.int64))


@dataset_transform
def transform_scaling(range=(0.0, 0.05), **kwargs):
    """
    Given an input of NxF, random scaling the coordinate. That is, transform the first 3 features but remains the
    other features unchanged
    :param range: The random scaling range
    :return: A function (NxF, _) -> (NxF, _), where the return point features have been transformed.
    "_" indicates the label input will remain unchanged.
    """
    return lambda points, label: (
        tf.concat(
            [points[..., :3] * tf.random.uniform((), minval=range[0], maxval=range[1]), points[..., 3:]],
            axis=-1
        ),
        label
    )


@dataset_transform
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
    def _transform_rotation(points, policy, range):
        if policy == "euler":
            mat = transforms3d.euler.euler2mat(
                random.uniform(range[0][0], range[0][1]),
                random.uniform(range[1][0], range[1][1]),
                random.uniform(range[2][0], range[2][1])
            )

            points = points.numpy()
            return np.concatenate(
                [np.matmul(mat, points[:, :3, np.newaxis])[..., 0], points[:, 3:]],
                axis=-1
            )
        else:
            assert False, "Quaternion is currently not supported"

    _transform_rotation_closoure = lambda points, label: (_transform_rotation(points, policy, range), label)
    return lambda points, label: tf.py_function(_transform_rotation_closoure, inp=[points, label], Tout=(tf.float32, tf.int64))

