from os import path
import tensorflow as tf
import logger


class ModelCallback(tf.keras.callbacks.Callback):
    """
    Custom model callback for:
        1.Evaluating dataset during training
        2.Log information
        3.Save weights during training
        4.Save best weights parameter
    """

    def __init__(self, train_step, validation_step, train_dataset, test_dataset, batch_size, save_dir, log_step=1):
        """
        Initialize a model callback
        :param train_step: How many batch before the training should be stopped
        :param validation_step: How many step(batch) should a test dataset should be evaluated
        :param train_dataset: The train dataset for the model
        :param test_dataset: The test dataset for the model
        :param batch_size: The batch size for the model
        :param save_dir: The save directory for the model
        :param log_step: How many step(batch) should a log something to the console
        """
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
            self._on_logging(batch, logs)

        # Validation
        if self.validation_step is not None and batch > 0 and batch % self.validation_step == 0:
            self._on_validation(batch, logs)

        # Stopping
        if batch > 0 and batch >= self.train_step:
            self._on_stop(batch, logs)

    def on_test_batch_begin(self, batch, logs=None):
        pass

    def on_test_batch_end(self, batch, logs=None):
        pass

    def _on_logging(self, batch, logs):
        results_output = ", ".join(["{}:{}".format(key, value) for key, value in logs.items()])
        logger.log("On batch {}/{}, results:{{{}}}".format(batch, self.train_step, results_output))

    def _on_validation(self, batch, logs):
        logger.log("On batch {}/{}, BEGIN EVALUATION".format(batch, self.train_step), color="blue")

        # Get the validation results
        metrics_values = self.model.evaluate(self.test_dataset, verbose=0)
        results = {name: value for name, value in zip(self.model.metrics_names, metrics_values)}
        results_output = ", ".join(["{}:{}".format(name, value) for name, value in results.items()])
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
        logger.log("Save info to \"{}\"".format(self.info_save_path), prefix=False, color="yellow")
        with open(self.info_save_path, "w") as fout:
            infos = {"step": batch}
            # Add the last results to it
            infos.update({("last_" + key): value for key, value in results.items()})
            # Add the best results to it
            infos.update({("best_" + key): value for key, value in self.best_results.items()})

        logger.log("On batch {}/{}, END EVALUATION".format(batch, self.train_step), color="blue")

    def _on_stop(self, batch, logs):
        logger.log("On batch {}/{}, stop".format(batch, self.train_step))
        logger.log("Save checkpoint and exit")
        self.model.save_weights(self.latest_save_path)


class ExponentialBiasedDecay(tf.keras.optimizers.schedules.ExponentialDecay):
    """The exponential decay learning rate schedule with a bias offset"""

    def __init__(self, offset, initial_learning_rate, decay_steps, decay_rate, staircase=False, name=None):
        """
        By default the original exponential learning rate schedule uses the decayed learning rate
        "initial_learning_rate * decay_rate ^ (step / decay_steps)", we add an offset to the "step", which is:
        "initial_learning_rate * decay_rate ^ ((step + offset) / decay_steps)". Such biased decay is useful
        when you want to resume a task with step 0 while the original training process has already iterated
        "offset" times. In this case, you simply use this class as a replacement to the original exponential
        learning schedule and it works well
        :param offset: The step offset
        :param initial_learning_rate: A scalar `float32` or `float64` `Tensor` or a Python number.
        The initial learning rate.
        :param decay_steps: A scalar `int32` or `int64` `Tensor` or a Python number. Must be positive.
        See the decay computation above.
        :param decay_rate:  A scalar `float32` or `float64` `Tensor` or a Python number.
        The decay rate.
        :param staircase: If `True` decay the learning rate at discrete intervals
        :param name: Optional name of the operation.  Defaults to 'ExponentialDecay'.
        """
        super(ExponentialBiasedDecay, self).__init__(
            offset=offset,
            initial_learning_rate=initial_learning_rate,
            decay_steps=decay_steps,
            decay_rate=decay_rate,
            staircase=staircase,
            name=name
        )
        self._offset = offset

    def __call__(self, step):
        super(ExponentialBiasedDecay, self).__call__(self, step + self._offset)