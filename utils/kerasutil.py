from os import path
from time import time

import tensorflow as tf
import logger

from utils.ioutil import save_model_to_h5, load_model_from_h5


def _step_biased(func):
    """
    A decorator that accepts a function with definition "func(self, batch, logs, ...)" and decorate it by
    offsetting the parameter "batch" to "batch + self.step_offset".
    :param func: The function with definition like "func(self, batch, logs, ...)"
    :return: The decorated function
    """

    def _func(self, batch, logs, *args, **kwargs):
        func(self, batch + self.step_offset, logs, *args, **kwargs)

    return _func


class ModelCallback(tf.keras.callbacks.Callback):
    """
    Custom model callback for:
        1.Evaluating dataset during training
        2.Log information
        3.Save weights during training
        4.Save best weights parameter
    """

    def __init__(self, train_step, validation_step, train_dataset, test_dataset,
                 batch_size, save_dir, infos, step_offset, log_step=1):
        """
        Initialize a model callback
        :param train_step: How many batch before the training should be stopped
        :param validation_step: How many step(batch) should a test dataset should be evaluated
        :param train_dataset: The train dataset for the model
        :param test_dataset: The test dataset for the model
        :param batch_size: The batch size for the model
        :param save_dir: The save directory for the model
        :param infos: The previous trained information. It is a dict contains some value like "best_accuracy" and
        "best_loss". It is used to initialize the best result from previous trained model. For a newly created model,
        the value is an empty dictionary.
        :param step_offset: How many steps have trained when resuming task from the previous one
        :param log_step: How many step(batch) should a log something to the console
        """
        super(ModelCallback, self).__init__()
        self.validation_step = validation_step
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.train_step = train_step
        self.save_dir = save_dir
        self.latest_save_path = path.join(save_dir, "latest_save.h5")
        self.best_save_path = path.join(save_dir, "best_save.h5")
        self.info_save_path = path.join(save_dir, "info.txt")
        self.step_offset = step_offset
        self.log_step = log_step

        self.best_results = {k[5:]: v for k, v in infos.items() if k.startswith("best")}

        # Use to show the diff time for each log
        self._last_log_time = None
        self._has_loaded = False

    def on_train_begin(self, logs=None):
        if path.isfile(self.latest_save_path):
            logger.log(f"Find latest state in \"{self.latest_save_path}\", loading")
            load_model_from_h5(self.model, self.latest_save_path)
            logger.log(f"Load latest state from \"{self.latest_save_path}\"", color="green")

    @_step_biased
    def on_train_batch_begin(self, batch, logs=None):
        # Load the previous state)
        self.model.reset_metrics()

    @_step_biased
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
            self.model.stop_training = True

        # Reset metrics for next batch
        self.model.reset_metrics()

    def _on_logging(self, batch, logs):

        # Compute the diff time
        current_log_time = time()
        diff_time = 0.0 if self._last_log_time is None else (current_log_time - self._last_log_time) * 1000.0
        self._last_log_time = current_log_time

        results_output = ", ".join(["{}:{}".format(key, value) for key, value in logs.items()])
        logger.log("On batch {}/{}, results:{{{}}}, time: {}ms".format(batch, self.train_step,
                                                                       results_output, diff_time))

    def _on_validation(self, batch, logs):
        # Get the validation results
        #metrics_values = self.model.evaluate(self.test_dataset, verbose=0)
        metrics_values = self.model.test_on_batch(self.test_dataset)

        
        results = {name: value for name, value in zip(self.model.metrics_names, metrics_values)}
        results_output = ", ".join(["{}:{}".format(name, value) for name, value in results.items()])
        logger.log("Evaluating validation dataset results: {{{}}}".format(results_output), prefix=False, color="blue")

        # Update the best results, we assume that we have a accuracy metrics in it
        best_accuracy = 0.0 if not self.best_results else self.best_results["accuracy"]
        accuracy = results["accuracy"]
        if self.best_results is None or best_accuracy < accuracy:
            logger.log("Best accuracy update: {} --> {}".format(best_accuracy, accuracy), prefix=False, color="green")
            self.best_results = results
            # Save the best checkpoint
            logger.log("Save best checkpoint to \"{}\"".format(self.best_save_path), prefix=False, color="green")
            save_model_to_h5(self.model, self.best_save_path)

        # Save the latest checkpoint
        logger.log("Save latest checkpoint to \"{}\"".format(self.latest_save_path), prefix=False, color="yellow")
        save_model_to_h5(self.model, self.latest_save_path)

        # Save info log
        logger.log("Save info to \"{}\"".format(self.info_save_path), prefix=False, color="yellow")
        with open(self.info_save_path, "w") as fout:
            infos = {"step": batch}
            # Add the last results to it
            infos.update({("last_" + key): value for key, value in results.items()})
            # Add the best results to it
            infos.update({("best_" + key): value for key, value in self.best_results.items()})
            fout.write("\n".join(["{} = {}".format(k, v) for k, v in infos.items()]))  # Write the result

        logger.log("On batch {}/{}, END EVALUATION".format(batch, self.train_step), color="blue")
        

    def _on_stop(self, batch, logs):
        logger.log("On batch {}/{}, stop".format(batch, self.train_step))
        logger.log("Save checkpoint and exit")
        self._on_validation(batch, logs)