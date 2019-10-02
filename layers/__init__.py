from .common import FeatureReshapeLayer
from .xconv import XConvLayer, XConvPoolingLayer


def layer_from_config(layer_conf):
    """
    Create a layer from configuration
    :param layer_conf: The configuration dict, where it should have an "name" entry specified the name of
    the layer and other parameter to initialize the layer
    :return: A corresponding keras layer
    """
    layer_map = {
        "conv-xconv": XConvLayer,
        "pooling-xconv": XConvPoolingLayer,
        "feature-reshape": FeatureReshapeLayer
    }

    assert layer_conf["name"] in layer_map, "Did not find layer with name \"{}\"".format(layer_conf["name"])
    return layer_map[layer_conf["name"]](**layer_conf)