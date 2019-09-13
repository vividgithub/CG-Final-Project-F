import datetime
import inspect


def log(info, *, prefix=True, end='\n'):
    """
    Log information
    :param info: The item to log
    :param prefix: Whether to print a time format print
    :param end: Same to the end in "print" builtin function
    :return: None
    """
    time_string = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    info_with_prefix = "{} in \"{}\": {}".format(time_string, inspect.stack()[1].function, info) if prefix else info
    print(info_with_prefix, end=end)