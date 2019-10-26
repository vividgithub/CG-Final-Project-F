# Computation Utility
import tensorflow as tf
from collections.abc import Iterable
import logger


def commonprefix(m):
    """Given a list of pathnames, returns the longest common leading component"""
    if not m: return ''
    s1 = min(m)
    s2 = max(m)
    for i, c in enumerate(s1):
        if c != s2[i]:
            return s1[:i]
    return s1


class ComputationContextLayer:
    """
    A mock for tf.keras.layer.Layer to let it only be called inside the computation context. This is
    done by hiding the __call__ attribute and expose a "_call_by_computation_context" attribute
    """
    def __init__(self, layer):
        self._layer = layer

    def call_by_computation_context(self, *args, **kwargs):
        return self._layer(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        logger.log(f"Warning: calling computation context layer {self._layer} "
                   f"with args: {args} kwargs: {kwargs} without using computation context", color="yellow")
        return self.call_by_computation_context(*args, **kwargs)


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
        def call_layer(l, x):
            if hasattr(l, "call_by_computation_context"):
                return l.call_by_computation_context(x, *self.args, **self.kwargs)
            else:
                return l(x, *self.args, **self.kwargs)

        if not isinstance(layers, (list, tuple)):
            # Single layer, just call
            return call_layer(layers, inputs)
        else:
            # Flatten the layers
            layers = tf.nest.flatten(layers)
            outputs = inputs

            # Add a name scope for wrapping
            scope_name = commonprefix([layer.name for layer in layers]).strip("-/\\?_ \t\n") or "ComputationContext"
            with tf.name_scope(scope_name):
                for l in layers:
                    outputs = call_layer(l, outputs)

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