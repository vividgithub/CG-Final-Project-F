import layers
import tensorflow_core as tf
import math
from os import path, makedirs
import logger
from utils.kerasutil import ModelCallback


def layer_from_config(layer_conf, model_conf, data_conf):
    """
    Get the corresponding keras layer from configurations
    :param layer_conf: The layer configuration
    :param model_conf: The global model configuration, sometimes it is used to generate some
    special layer like "output-classification" and "output-segmentation" layer
    :param data_conf: The dataset configuration, for generating special layers
    :return: A keras layer
    """
    name = layer_conf["name"]

    if name == "output-classification" or name == "output-segmentation":
        return tf.keras.layers.Dense(data_conf["class_count"])
    elif name == "output-conditional-segmentation":
        return layers.OutputConditionalSegmentationLayer(data_conf["class_count"])
    else:
        return layers.layer_from_config(layer_conf)


def optimizer_from_config(learning_rate, optimizer_conf):
    """
    Get the optimizer from configuration
    :param learning_rate: The learning rate, might be a scalar or a learning rate schedule
    :param optimizer_conf: The optimizer configuration
    :return: An corresponding optimizer
    """
    optimizer_map = {
        "adam": tf.keras.optimizers.Adam
    }

    conf = optimizer_conf.copy()
    name = conf["name"]
    del conf["name"]

    return optimizer_map[name](learning_rate=learning_rate, **conf)


def learning_rate_from_config(learning_rate_conf):
    """
    Get the learning rate scheduler based on configuration
    :param learning_rate_conf: The learning rate configuration
    :return: A learning rate scheduler
    """
    learning_rate_map = {
        "exponential_decay": tf.keras.optimizers.schedules.ExponentialDecay
    }

    conf = learning_rate_conf.copy()
    name = conf["name"]
    del conf["name"]

    return learning_rate_map[name](**conf)


def net_from_config(model_conf, data_conf):
    """
    Generate a keras network from configuration dict
    :param model_conf: The global model configuration dictionary
    :param data_conf: The configuration of the dataset, it might use to initialize some layer like
    "output-classification"
    :param train_dataset: The train dataset, used to add input layer based on shape
    :return: A keras net
    """
    # Get network conf
    net_conf = model_conf["net"]

    # Input layer
    transform_confs = model_conf["dataset"].get("train_transforms", [])
    # Get the shape of the dataset, first check whether we have clip-feature layer in the dataset, if not, we
    # use the feature size in the dataset configuration
    feature_size = None
    for transform_conf in transform_confs[::-1]:
        if type(transform_conf) is dict and transform_conf.get("name") == "clip-feature":
            feature_size = transform_conf["c"]
            logger.log("Get feature_size={} from model configuration".format(feature_size))
    if feature_size is None:
        feature_size = data_conf.get("feature_size")
        logger.log("Get feature_size={} from dataset configuration".format(feature_size))
    assert feature_size is not None, "Cannot determine the feature_size"
    # Get the point size, if possible
    point_count = data_conf.get("point_count")
    for transform_conf in transform_confs[::-1]:
        if type(transform_conf) is dict and transform_conf.get("name") == "sampling":
            point_count = None
            logger.log("Ignore point_count since we have transform sampling from dataset")
    # input_layer = tf.keras.layers.InputLayer(input_shape=(point_count, feature_size))

    # Extend feature layer
    extend_feature_layer = None
    if "extend_feature" in net_conf:
        if net_conf["extend_feature"] == "none":
            pass
        else:
            # TODO: Extend feature
            assert False, "Other extend feature not implemented"

    if net_conf["structure"] == "sequence":
        x = inputs = tf.keras.Input(shape=(point_count, feature_size))  # Input layer
        x = layers.common.DataSplitLayer()(x)  # Split data

        # Add extend feature layer
        if extend_feature_layer:
            x = extend_feature_layer(x)

        for layer_conf in net_conf["layers"]:
            layer = layer_from_config(layer_conf, model_conf, data_conf)
            x = layer(x)

        outputs = x
        return tf.keras.Model(inputs=inputs, outputs=outputs)
    else:
        assert False, "\"{}\" is currently not supported".format(net_conf["structure"])


class ModelRunner:
    """
    A class to run a specified model on a specified dataset
    """

    def __init__(self, model_conf, data_conf, name, save_root_dir, train_dataset, test_dataset):
        """
        Initialize a model runner
        :param model_conf: The pyconf for model
        :param data_conf: The pyconf for dataset
        :param name: The name for model
        :param save_root_dir: The root for saving. Normally it is the root directory where all the models of a specified
        dataset should be saved. Like something "path/ModelNet40-2048". Note that it is not the "root directory of the
        model", such as "path/ModelNet40-2048/PointCNN-X3-L4".
        :param train_dataset: The dataset to train the model
        :param test_dataset: The dataset to test the model
        """
        self.model_conf = model_conf
        self.data_conf = data_conf
        self.name = name
        self.save_root_dir = save_root_dir
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

    def train(self):
        # TODO: Add mode
        control_conf = self.model_conf["control"]

        # Transform the dataset is the dataset is classification dataset and
        # the model_conf's last output layer is output-conditional-segmentation
        train_dataset = test_dataset = None
        if self.data_conf["task"] == "classification" and \
                self.model_conf["net"]["layers"][-1]["name"] == "output-conditional-segmentation":
            layer_conf = self.model_conf["net"]["layers"][-1]
            assert "output_size" in layer_conf, "The dataset is classification dataset " \
                                                "while the model configuration is segmentation. " \
                                                "Cannot find \"output_size\" to transform the " \
                                                "classification dataset to segmentation task"
            seg_output_size = layer_conf["output_size"]
            # Transform function convert the label with (B, 1) to (B, N) where N is the last layer's point output size
            transform_func = (lambda points, label: (points, tf.tile(label, (1, seg_output_size))))
            train_dataset = self.train_dataset.map(transform_func)
            test_dataset = self.test_dataset
            logger.log("Convert classification to segmentation task with output_size={}".format(seg_output_size))

        # Get the network
        logger.log("Creating network, train_dataset={}, test_dataset={}".format(self.train_dataset, self.test_dataset))
        net = net_from_config(self.model_conf, self.data_conf)

        # Get the learning_rate and optimizer
        logger.log("Creating learning rate schedule")
        lr_schedule = learning_rate_from_config(control_conf["learning_rate"])
        logger.log("Creating optimizer")
        optimizer = optimizer_from_config(lr_schedule, control_conf["optimizer"])

        # Get the loss
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, name="Loss")

        # Get the metrics
        # We add a logits loss in the metrics since the total loss will have regularization term
        metrics = [
            tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.SparseCategoricalCrossentropy(from_logits=True, name="logits_loss")
        ]

        # Get the batch size
        batch_size = control_conf["batch_size"]

        # Get the total step for training
        if "train_epoch" in control_conf:
            train_step = int(math.ceil(control_conf["train_epoch"] * self.data_conf["train"]["size"] / batch_size))
        elif "train_step" in control_conf:
            train_step = control_conf["train_step"]
        else:
            assert False, "Do not set the \"train_step\" or \"train_epoch\" in model configuraiton"

        # Get the validation step
        validation_step = control_conf.get("validation_step", None)
        tensorboard_sync_step = control_conf.get("tensorboard_sync_step", None) or validation_step or 100

        logger.log("Training conf: batch_size={}, train_step={}, validation_step={}, "
                   "tensorboard_sync_step={}".format(batch_size, train_step, validation_step, tensorboard_sync_step))

        # Get the save directory
        suffix = 0
        while True:
            model_name_with_suffix = self.name + ("-" + str(suffix) if suffix > 0 else "")
            save_dir = path.join(self.save_root_dir, model_name_with_suffix)
            if not path.exists(save_dir):
                break
            suffix += 1
        makedirs(save_dir, exist_ok=False)
        logger.log("Save in directory: \"{}\"".format(save_dir))

        # Get the callback
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=save_dir, update_freq=tensorboard_sync_step)
        model_callback = ModelCallback(train_step, validation_step, train_dataset, test_dataset, batch_size, save_dir)

        logger.log("Compile network, loss={}, metrics={}".format(loss, metrics))
        net.compile(optimizer, loss=loss, metrics=metrics)

        logger.log("Summary of the network:")
        net.summary(line_length=240, print_fn=lambda x: logger.log(x, prefix=False))

        logger.log("Begin training")
        net.fit(
            train_dataset,
            verbose=0,
            steps_per_epoch=train_step,
            callbacks=[tensorboard_callback, model_callback],
            shuffle=False  # We do the shuffle ourself
        )