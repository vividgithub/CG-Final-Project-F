from os import path
import h5py


def pyconf(filepath):
    """
    Load a *.pyconf from specified file path
    :param filepath: The file path for pyconf
    :return: A dictionary for the configuration
    """
    assert path.exists(filepath) and not path.isdir(filepath), "{} doesn't exist or it is a directory".format(filepath)
    with open(filepath) as f:
        conf = eval(f.read())

    # Resolve reference link and evaluation parameter
    def ref_link(c, g):
        if type(c) is str and c.startswith("@"):  # Reference link, something link "@dataset/name"
            ref_path = c.strip()[1:].split("/")
            c = g
            for ref in ref_path:
                # We only support the indexing of dictionary and list (including tuple)
                c = c[ref] if type(c) is dict else c[int(ref)]
            return c
        elif type(c) is str and c.startswith("$"):  # Evaluation parameter
            return eval(c[1:])
        elif type(c) is dict:  # Dictionary
            return {k: ref_link(v, g) for k, v in c.items()}
        elif type(c) is list or type(c) is tuple:  # List or tuple
            return type(c)((ref_link(x, g) for x in c))
        else:  # Element
            return c

    conf = ref_link(conf, conf)
    return conf


def save_model_to_h5(model, path):
    """
    Save a model's weights and optimizers' weights to a single h5 file
    :param model: The model to save
    :param path: The path for saving
    :return: None
    """
    from tensorflow_core.python.keras.saving.hdf5_format import save_weights_to_hdf5_group
    from tensorflow_core.python.keras.saving.hdf5_format import save_optimizer_weights_to_hdf5_group

    with h5py.File(path, mode='w') as f:
        # Save models weight
        model_weights_group = f.create_group("model_weights")
        save_weights_to_hdf5_group(model_weights_group, model.layers)

        # Save optimizers weight
        save_optimizer_weights_to_hdf5_group(f, model.optimizer)

        f.flush()


def load_model_from_h5(model, path):
    """
    Load a model's weights and optimizers' weights from a single h5 file
    :param model: The model to load
    :param path: The path to load the model
    :return: A loaded model
    """
    from tensorflow_core.python.keras.saving.hdf5_format import load_weights_from_hdf5_group
    from tensorflow_core.python.keras.saving.hdf5_format import load_optimizer_weights_from_hdf5_group

    with h5py.File(path, mode="r") as f:
        # Load weights
        load_weights_from_hdf5_group(f["model_weights"], model.layers)
        optimizer_weight_values = load_optimizer_weights_from_hdf5_group(f)
        try:
            # Build train function (to get weight updates).
            # Models that aren't graph networks must wait until they are called
            # with data to _make_train_function() and so can't load optimizer
            # weights.
            model._make_train_function()
            model.optimizer.set_weights(optimizer_weight_values)
        except ValueError:
            raise
