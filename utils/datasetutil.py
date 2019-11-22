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

    train_list_filename = data_conf.get("type", dict()).get("train_file_list")
    test_list_filename = data_conf.get("type", dict()).get("test_file_list")

    train_list_filepath = path.join(dir, train_list_filename)
    test_list_filepath = path.join(dir, test_list_filename)
    log("Loading train file=\"{}\", policy={}".format(train_list_filepath, train_load_policy))
    log("Loading test file \"{}\", policy={}".format(test_list_filepath, test_load_policy))

    def _h5_generator(filepath, policy):
        for h5_filename in train_file_list:
            print(h5_filename)
            f = h5py.File(h5_filename)
            try:
                data = f['data']
                group = f['pid']
            except:
                pass
            if 'label' in f:
                label = f['label']
            else:
                label = []
            if 'seglabel' in f:
                seg = f['seglabel']
            else:
                seg = f['seglabels']
            for points, group_label, sem_label in zip(data, group, seg):
                label = []
                label.append(group)
                label.append(seg)
                label = np.array(label)
                yield points, label

    train_file_list = [line.rstrip() for line in open(train_list_filepath)]
    test_file_list = [line.rstrip() for line in open(test_list_filepath)]

    train_gen = lambda: _h5_generator(train_file_list)
    test_gen = lambda: _h5_generator(test_file_list)
    output_types = (tf.float32, tf.int32)

    point_count = data_conf.get("point_count", None)
    feature_size = data_conf.get("feature_size", None)
    output_shapes = ((point_count, feature_size), (2,))

    return (
        tf.data.Dataset.from_generator(train_gen, output_types=output_types, output_shapes=output_shapes),
        tf.data.Dataset.from_generator(test_gen, output_types=output_types, output_shapes=output_shapes)
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

    train_dataset = apply_transforms(train_dataset, model_conf["dataset"].get("train_transforms", []), batch_size)
    test_dataset = apply_transforms(test_dataset, model_conf["dataset"].get("test_transforms", []), batch_size)

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


@register_conf(name="shuffle", scope="transform", conf_func="self")
@dataset_transform
def transform_shuffle(buffer_size=4096, **kwargs):
    """
    A wrapper for dataset shuffle
    :param batch_size: The provided batch size
    :param buffer_multiplier: A multiplier to the batch size.
    The shuffle the buffer size is equal to "batch_size * buffer_multiplier"
    :return: A transform function
    """

    return lambda dataset: dataset.shuffle(buffer_size=buffer_size)


@register_conf(name="repeat", scope="transform", conf_func="self")
@dataset_transform
def transform_repeat(**kwargs):
    """
    A wrapper for dataset repeat
    :return: A transform function
    """
    return lambda dataset: dataset.repeat()
