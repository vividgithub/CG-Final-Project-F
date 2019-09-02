import datetime
import inspect


def log(info):
    """
    Log information
    :param info: The string to log
    :return: None
    """
    time_string = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    format = "{} in \"{}\": {}".format(time_string, inspect.stack()[1].function, info)
    print(format)