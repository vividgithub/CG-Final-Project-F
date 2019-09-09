import layers
import tensorflow as tf


def net_from_config(model_conf, data_conf):
    """
    Generate a keras network from configuration dict
    :param model_conf: The global model configuration dictionary
    :param data_conf: The configuration of the dataset, it might use to initialize some layer like
    "output-classification"
    :return: A keras model
    """

    def layer_func(layer_conf):
        special_layer_map = {
            "output-classification": (lambda _: tf.keras.layers.Dense(data_conf["class_count"], activation="softmax"))
        }
        if layer_conf["name"] in special_layer_map:
            return special_layer_map[layer_conf["name"]](layer_conf)
        else:
            return layers.layer_from_config(layer_conf)

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
            net.add(layer_func(layer_conf))

        return net
    else:
        assert False, "\"{}\" is currently not supported".format(net_conf["structure"])