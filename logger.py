import datetime
import inspect

_color_code = {
    "red": "\x1b[31m",
    "green": "\x1b[32m",
    "yellow": "\x1b[33m",
    "blue": "\x1b[34m"
}


def log(info, *, prefix=True, end='\n', color=None):
    """
    Log information
    :param info: The item to log
    :param prefix: Whether to print a time format print
    :param end: Same to the end in "print" builtin function
    :param color: The decorated color to print with, only works for std output
    :return: None
    """
    time_string = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

    info_with_prefix = "{} in \"{}\": {}".format(time_string, inspect.stack()[1].function, info) if prefix else info
    info_with_prefix_colored = _color_code.get(color, "") + info_with_prefix + "\x1b[0m" if color is not None else info_with_prefix

    print(info_with_prefix_colored, end=end)