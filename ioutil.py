from os import path


def pyconf(filepath):
    """
    Load a *.pyconf from specified file path
    :param filepath: The file path for pyconf
    :return: A dictionary for the configuration
    """
    assert path.exists(filepath) and not path.isdir(filepath), "{} doesn't exist or it is a directory".format(filepath)
    with open(filepath) as f:
        return eval(f.read())