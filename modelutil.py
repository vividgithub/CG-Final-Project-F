import layers
import tensorflow_core as tf
import math
from os import path, makedirs
import logger


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
    input_layer = tf.keras.layers.InputLayer(input_shape=(point_count, feature_size))

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

        # Add input layer
        net.add(input_layer)

        # Add extend feature layer
        if extend_feature_layer:
            net.add(extend_feature_layer)

        # Add another
        for layer_conf in net_conf["layers"]:
            net.add(layer_from_config(layer_conf, model_conf, data_conf))

        return net
    else:
        assert False, "\"{}\" is currently not supported".format(net_conf["structure"])


# def loss_from_config(model_config, data_config):
#     """
#     Get the loss from model_config and data_config
#     :param model_config: The global model configuration
#     :param data_config: The data configuration
#     :return: A loss function
#     """
#
#     def sparse_categorical_classification_to_segmentation_cross_entropy(y_true, y_pred, from_logits=False, axis=-1):
#         # Compare the loss of y_true.shape = (B,) with y_pred.shape = (B, N, C)
#         return tf.keras.losses.sparse_categorical_crossentropy(
#             y_true=K.tile(y_true, (1, tf.shape(y_pred)[1])),
#             y_pred=y_pred,
#             from_logits=from_logits,
#             axis=axis
#         )
#
#     # Normally this function will return SparseCategoricalCrossentropy except that we use
#     # a classification dataset but the network's last layer is "output-segmentation", which
#     # indicates that the dataset shape is (B, ) but the output shape is (B, N, C). So we have
#     # to use a special loss function to such case
#     if data_config["task"] == "classification" and model_config["net"]["layers"][-1]["name"] == "output-segmentation":
#         return SparseCategoricalClassificationToSegmentationCrossEntropy()
#     else:
#         return tf.keras.losses.SparseCategoricalCrossentropy()
#
#
# def metrics_from_config(model_config, data_config):
#     """
#     Get the corresponding default metrics from model configuration and data configuration
#     :param model_config: The global model configuration
#     :param data_config: The dataset configuration
#     :return: A group of metrics
#     """
#     def sparse_categorical_classification_to_segmentation_accuracy(y_true, y_pred):
#         # Compare the loss of y_true.shape = (B, 1) with y_pred.shape = (B, N, C)
#
#         # Normally, when we don't specify the target_tensors in "model.compile", keras will assume
#         # that the y_true has the same dimension shape of y_pred, if this case, it is (None, None, None).
#         # So in order to replicate the y_true for N times, we need to first flatten it to a 1D-tensor and
#         # then tile it with N.
#         y_true = tf.reshape(y_true, (-1, ))  # (None, None, None) -> (None, )
#         y_true = tf.tile(y_true, (tf.shape(y_pred)[1], ))  # (None, ) -> (None, )
#         y_true = tf.reshape(y_true, (tf.shape(y_pred)[0], -1))  # (None, ) -> (None, None)
#
#         return tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)
#
#     # Same as "sparse_categorical_classification_to_segmentation_cross_entropy", when a classification dataset
#     # is used but the network's last layer is "output-segmentation" (which implies the shape is (B, N, F)), we
#     # have to define a new metrics for classification for label input (B,)
#     if data_config["task"] == "classification" and model_config["net"]["layers"][-1]["name"] == "output-segmentation":
#         return [MeanMetricWrapper(sparse_categorical_classification_to_segmentation_accuracy)]
#     else:
#         return [tf.keras.metrics.SparseCategoricalAccuracy()]


class ModelCallback(tf.keras.callbacks.Callback):

    def __init__(self, train_step, validation_step, test_dataset, batch_size, save_dir, log_step=None):
        self.validation_step = validation_step
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.train_step = train_step
        self.save_dir = save_dir
        self.latest_save_path = path.join(save_dir, "latest_save.h5")
        self.best_save_path = path.join(save_dir, "best_save.h5")
        self.log_step = log_step

    def on_train_batch_begin(self, batch, logs=None):
        pass

    def on_train_batch_end(self, batch, logs=None):
        # Log
        if self.log_step and batch % self.log_step == 0:
            logger.log("On batch {}/{}, loss={}".format(batch, self.train_step, logs["loss"]))

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

        results = self.model.evaluate(self.test_dataset, verbose=0)
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

    # Transform the dataset is the dataset is classification dataset and
    # the model_conf's last output layer is output-segmentation
    if data_config["task"] == "classification" and model_config["net"]["layers"][-1]["name"] == "output-segmentation":
        layer_conf = model_config["net"]["layers"][-1]
        assert "output_size" in layer_conf, "The dataset is classification dataset " \
                                            "while the model configuration is segmentation. " \
                                            "Cannot find \"output_size\" to transform the " \
                                            "classification dataset to segmentation task"
        seg_output_size = layer_conf["output_size"]
        # Transform function convert the label with (B, 1) to (B, N) where N is the last layer's point output size
        transform_func = (lambda points, label: (points, tf.tile(label, (1, seg_output_size))))
        train_dataset = train_dataset.map(transform_func)
        test_dataset = test_dataset.map(transform_func)
        logger.log("Convert classification to segmentation task with output_size={}".format(seg_output_size))

    # Get the network
    logger.log("Creating network, train_dataset={}, test_dataset={}".format(train_dataset, test_dataset))
    net = net_from_config(model_config, data_config)

    # Get the learning_rate and optimizer
    logger.log("Creating learning rate schedule")
    lr_schedule = learning_rate_from_config(control_conf["learning_rate"])
    logger.log("Creating optimizer")
    optimizer = optimizer_from_config(lr_schedule, control_conf["optimizer"])

    # Get the loss
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # Get the metrics
    metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]

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
        suffix += 1
    makedirs(save_dir, exist_ok=False)
    logger.log("Save in directory: \"{}\"".format(save_dir))

    # Get the callback
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=save_dir, update_freq=tensorboard_sync_step)
    model_callback = ModelCallback(train_step, validation_step, test_dataset, batch_size, save_dir)

    logger.log("Compile network, loss={}, metrics={}".format(loss, metrics))
    net.compile(optimizer, loss=loss, metrics=metrics)

    logger.log("Summary of the network:")
    net.summary(print_fn=lambda x: logger.log(x, prefix=False))

    logger.log("Begin training")
    net.fit(
        train_dataset,
        verbose=1,
        steps_per_epoch=train_step,
        callbacks=[tensorboard_callback, model_callback],
        shuffle=False  # We do the shuffle ourself
    )