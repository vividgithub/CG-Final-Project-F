import tensorflow as tf
from utils.computil import ComputationContextLayer


class ComposeLayer(tf.keras.layers.Layer):
    """
    A layer that uses other layer as sublayer for assistant computation
    """
    def __init__(self, *args, **kwargs):
        super(ComposeLayer, self).__init__(*args, **kwargs)
        self.sub_layers = dict()
        self._c = None

    def add_layer(self, name, layer_type, *args, **kwargs):
        """
        Add a layer as its sublayer for computation if the unique name of the layer if not exists
        :param name: An unique name of that layer
        :param layer_type: The class of the layer
        :param args: The args for initialization of the layer
        :param kwargs: The kwargs for initialization of the layer
        :return: An instance of that layer
        """
        if self.sub_layers.get(name) is None:
            self.sub_layers[name] = ComputationContextLayer(layer_type(*args, **kwargs))
        return self.sub_layers[name]
