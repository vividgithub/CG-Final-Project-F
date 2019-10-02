from os import path


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
                c = g[ref] if type(g) is dict else g[int(ref)]
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
