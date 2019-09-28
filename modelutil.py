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


class ModelCallback(tf.keras.callbacks.Callback):

    def __init__(self, train_step, validation_step, train_dataset, test_dataset, batch_size, save_dir, log_step=1):
        self.validation_step = validation_step
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.train_step = train_step
        self.save_dir = save_dir
        self.latest_save_path = path.join(save_dir, "latest_save.h5")
        self.best_save_path = path.join(save_dir, "best_save.h5")
        self.info_save_path = path.join(save_dir, "info.txt")
        self.log_step = log_step

        self.best_results = None

    def on_train_batch_begin(self, batch, logs=None):
        self.model.reset_metrics()

    def on_train_batch_end(self, batch, logs=None):
        # Log
        if self.log_step and batch % self.log_step == 0:
            self.on_logging(batch, logs)

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

    def on_logging(self, batch, logs):
        results_output = ", ".join(["{}:{}".format(key, value) for key, value in logs.items()])
        logger.log("On batch {}/{}, results:{{{}}}".format(batch, self.train_step, results_output))

    def on_validation(self, batch, logs):
        logger.log("On batch {}/{}, BEGIN EVALUATION".format(batch, self.train_step), color="blue")

        # Get the validation results
        metrics_values = self.model.evaluate(self.test_dataset, verbose=0)
        results = {name: value for name, value in zip(self.model.metrics_names, metrics_values)}
        results_output = ", ".join(["{}:{}".format(name, value) for name, value in results.items()])
        logger.log(results_output, prefix=False)
        logger.log("Evaluating validation dataset results: {{{}}}".format(results_output), prefix=False, color="blue")

        # Update the best results, we assume that we have a accuracy metrics in it
        best_accuracy = 0.0 if self.best_results is None else self.best_results["accuracy"]
        accuracy = results["accuracy"]
        if self.best_results is None or best_accuracy < accuracy:
            logger.log("Best accuracy update: {} --> {}".format(best_accuracy, accuracy), prefix=False, color="green")
            self.best_results = results
            # Save the best checkpoint
            logger.log("Save best checkpoint to \"{}\"".format(self.best_save_path), prefix=False, color="green")
            self.model.save_weights(self.best_save_path)

        # Save the latest checkpoint
        logger.log("Save latest checkpoint to \"{}\"".format(self.latest_save_path), prefix=False, color="yellow")
        self.model.save_weights(self.latest_save_path)

        # Save info log
        logger.log("Save info to \"{}\"".format(self.info_save_path), prefix=False)
        with open(self.info_save_path, "w") as fout:
            infos = {"step": batch}
            # Add the last results to it
            infos.update({("last_" + key): value for key, value in results.items()})
            # Add the best results to it
            infos.update({("best_" + key): value for key, value in self.best_results.items()})

        logger.log("On batch {}/{}, END EVALUATION".format(batch, self.train_step), color="blue")

    def on_stop(self, batch, logs):
        logger.log("On batch {}/{}, stop".format(batch, self.train_step))
        logger.log("Save checkpoint and exit")
        self.model.save_weights(self.latest_save_path)


def train_model(model_config, data_config, model_name, save_root_dir, train_dataset, test_dataset):
    # TODO: Add mode
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
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, name="Loss")

    # Get the metrics
    # We add a logits loss in the metrics since the total loss will have regularization term
    metrics = [
        tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
        tf.keras.metrics.SparseCategoricalCrossentropy(from_logits=True, name="logits-loss")
    ]

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
    model_callback = ModelCallback(train_step, validation_step, train_dataset, test_dataset, batch_size, save_dir)

    logger.log("Compile network, loss={}, metrics={}".format(loss, metrics))
    net.compile(optimizer, loss=loss, metrics=metrics)

    logger.log("Summary of the network:")
    net.summary(print_fn=lambda x: logger.log(x, prefix=False))

    logger.log("Begin training")
    net.fit(
        train_dataset,
        verbose=0,
        steps_per_epoch=train_step,
        callbacks=[tensorboard_callback, model_callback],
        shuffle=False  # We do the shuffle ourself
    )