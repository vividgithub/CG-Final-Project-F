import layers
import tensorflow as tf
import math
from os import path, makedirs
import logger
from tensorflow_core.python.keras.losses import LossFunctionWrapper


def layer_from_config(layer_conf, model_conf, data_conf):
    """
    Get the corresponding keras layer from configurations
    :param layer_conf: The layer configuration
    :param model_conf: The global model configuration, sometimes it is used to generate some
    special layer like "output-classification" and "output-segmentation" layer
    :param data_conf: The dataset configuration, for generating special layers
    :return: A keras layer
    """
    # Delete name
    name = layer_conf["name"]
    conf = layer_conf.copy()
    del conf["name"]

    if name == "output-classification" or name == "output-segmentation":
        return tf.keras.layers.Dense(data_conf["class_count"])
    else:
        return layers.layer_from_config(conf)


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
    :return: A keras net
    """
    # Get network conf
    net_conf = model_conf["net"]

    # Extend feature layer
    extend_feature_layer = None
    if "extend_feature" in net_conf:
        if net_conf["extend_feature"] == "none":
            pass
        else:
            # TODO: Extend feature
            assert False, "Other extend feature not implemented"

    if net_conf["structure"] == "sequence":
        # Generate sequence network
        net = tf.keras.Sequential()

        # Add extend feature layer
        if extend_feature_layer:
            net.add(extend_feature_layer)

        # Add another
        for layer_conf in net_conf["layers"]:
            net.add(layer_from_config(layer_conf, model_conf, data_conf))

        return net
    else:
        assert False, "\"{}\" is currently not supported".format(net_conf["structure"])


def loss_from_config(model_config, data_config):
    """
    Get the loss from model_config and data_config
    :param model_config: The global model configuration
    :param data_config: The data configuration
    :return: A loss function
    """

    def sparse_categorical_classification_to_segmentation_cross_entropy(y_true, y_pred, from_logits=False, axis=-1):
        # Compare the loss of y_true.shape = (B,) with y_pred.shape = (B, N, C)
        return tf.keras.losses.sparse_categorical_crossentropy(
            y_true=tf.tile(y_true, (1, tf.shape(y_pred)[1])),
            y_pred=y_pred,
            from_logits=from_logits,
            axis=axis
        )

    # Normally this function will return SparseCategoricalCrossentropy except that we use
    # a classification dataset but the network's last layer is "output-segmentation", which
    # indicates that the dataset shape is (B, ) but the output shape is (B, N, C). So we have
    # to use a special loss function to such case
    if data_config["task"] == "classification" and model_config["net"]["layers"][-1]["name"] == "output-segmentation":
        return LossFunctionWrapper(fn=sparse_categorical_classification_to_segmentation_cross_entropy)
    else:
        return tf.keras.losses.SparseCategoricalCrossentropy()


class ModelCallback(tf.keras.callbacks.Callback):

    def __init__(self, train_step, validation_step, test_dataset, batch_size, save_dir):
        self.validation_step = validation_step
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.train_step = train_step
        self.save_dir = save_dir
        self.latest_save_path = path.join(save_dir, "latest_save.h5")
        self.best_save_path = path.join(save_dir, "best_save.h5")

    def on_train_batch_begin(self, batch, logs=None):
        pass

    def on_train_batch_end(self, batch, logs=None):
        # Validation
        if self.validation_step is not None and batch > 0 and batch % self.validation_step == 0:
            self.on_validation(batch, logs)

        # Stopping
        if batch > 0 and batch >= self.train_step:
            self.on_stop(batch, logs)

    def on_test_batch_begin(self, batch, logs=None):
        pass

    def on_test_batch_end(self, batch, logs=None):
        pass

    def on_validation(self, batch, logs):
        logger.log("On batch {}/{}, begin evaluation".format(batch, self.train_step))

        results = self.model.evaluate(self.test_dataset, batch_size=self.batch_size, verbose=0)
        logger.log("Evaluation result:\n{}".format(batch, self.train_step, results))

        logger.log("Save checkpoint and reset metrics")
        self.model.save_weights(self.latest_save_path)
        self.model.reset_metrics()

        # TODO: Save best

        logger.log("On batch {}/{}, end evaluation".format(batch, self.train_step))

    def on_stop(self, batch, logs):
        logger.log("On batch {}/{}, stop".format(batch, self.train_step))
        logger.log("Save checkpoint and exit")
        self.model.save_weights(self.latest_save_path)


def train_model(model_config, data_config, model_name, save_root_dir, train_dataset, test_dataset):
    control_conf = model_config["control"]

    # Get the network
    logger.log("Creating network")
    net = net_from_config(model_config, data_config)

    # Get the learning_rate and optimizer
    logger.log("Creating learning rate schedule")
    lr_schedule = learning_rate_from_config(control_conf["learning_rate"])
    logger.log("Creating optimizer")
    optimizer = optimizer_from_config(lr_schedule, control_conf["optimizer"])

    # Get the loss
    loss = loss_from_config(model_config, data_config)

    # Get the batch size
    batch_size = control_conf["batch_size"]

    # Get the total step for training
    if "train_epoch" in control_conf:
        train_step = int(math.ceil(control_conf["train_epoch"] * data_config["train"]["size"] / batch_size))
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
        model_name_with_suffix = model_name + ("-" + str(suffix) if suffix > 0 else "")
        save_dir = path.join(save_root_dir, model_name_with_suffix)
        if not path.exists(save_dir):
            break
    makedirs(save_dir, exist_ok=False)
    logger.log("Save directory: \"{}\"".format(save_dir))

    # Get the callback
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=save_dir, update_freq=tensorboard_sync_step)
    model_callback = ModelCallback(train_step, validation_step, test_dataset, batch_size, save_dir)

    net.compile(optimizer, loss=loss, metrics=None)  # TODO: SparseCategoricalClassificationToSegmentationAccuracy
    net.fit(
        train_dataset,
        batch_size=batch_size,
        verbose=1,
        callbacks=[tensorboard_callback, model_callback],
        shuffle=False  # We do the shuffle ourself
    )