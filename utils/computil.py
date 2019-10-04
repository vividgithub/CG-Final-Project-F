# Computation Utility
import tensorflow as tf
from collections.abc import Iterable


def flatten(items):
    """Yield items from any nested iterable; see Reference."""
    for x in items:
        if isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            for sub_x in flatten(x):
                yield sub_x
        else:
            yield x


class ComputationContext:
    """
    A syntax sugar class for simplification for computation in "call" method. When you try to do something
    like "x = sublayer1(x, **kwargs); x = sublayer2(x, **kwargs)...", you only need to do
    "c = ComputationContext(**kwargs); x = c((sublayer1, sublayer2), x)"
    """
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __call__(self, layers, inputs):
        """
        Calling a ComputationContext with a single layer or a nested list of layer object, it will call them
        sequentially.
        :param layers: The layer(s) to be called, it can be a single layer object, or a NESTED list of layers
        :param inputs: The inputs
        :return: The outputs after the layer(s)
        """
        outputs = inputs
        # if isinstance(layers, Iterable):
        #     for layer in flatten(layers):
        #         outputs = layer(outputs, *self.args, **self.kwargs)
        # else:
        #     outputs = layers(outputs)
        for layer in layers:
            outputs = layer(outputs, *self.args, **self.kwargs)
        return outputs


class CurryingWrapper:
    """
    A CurryingWrapper object takes a function and its args and convert it to a function that only call with the
    first argument. like CurryWrapper(func, ...)(x) is equal to calling func(x, ...)
    """
    def __init__(self, func, *args, **kwargs):
        """
        Initialization
        :param func: The function to wrap
        :param args: The args for function (start from second)
        :param kwargs: The kwargs for function
        """
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def __call__(self, x, *args, **kwargs):
        return self.func(x, *self.args, self.kwargs)


def _(func, *args, **kwargs):
    """
    A alias for calling CurryingWrapper(func, *args, **kwargs)
    :param func: The function to wrap
    :param args: The args for function (start from second)
    :param kwargs: The kwargs for function
    :return: A CurryingWrapper object
    """
    return CurryingWrapper(func, *args, **kwargs)