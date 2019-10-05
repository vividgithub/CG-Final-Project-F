class ConfigManager:
    """
    A configuration manager that manages all configuration de-serialization works. That is,
    how to register a function to initialize an object and how to get object from configuration
    """
    SCOPES = {"transform", "layer", "op", "learning_rate", "optimizer"}

    def __init__(self):
        """
        Initialize a configuration manager
        """
        # Flat map maps the name to an object (which is configurable)
        # Map maps a tuple (scope, name) to an object
        # Map is unique for (scope, name) key but flat map may contain conflicts
        # (the same name with difference scope). But sometimes it is useful to just
        # query from flat_map if there's no conflict happens
        self.flat_map = dict()
        self.map = dict()

    def register_conf(self, scope, name, conf_func):
        """
        Adds an entry to the configuration manager
        :param scope: The scope to search, should be in SCOPES
        :param name: The name to register
        :param conf_func: A function that takes a configuration dictionary as parameter
        and returns an instance of that object
        :return: None
        """
        assert callable(conf_func), f"The registed configuration function is not callable"
        assert (scope, name) not in self.map, \
            f"The \"{name}\" has already been registered in scope \"{scope}\", which is \"{self.map[(scope, name)]}\""
        assert scope in ConfigManager.SCOPES, f"\"{scope}\" is not in {ConfigManager.SCOPES}"

        self.flat_map.setdefault(name, []).append(conf_func)
        self.map[(scope, name)] = conf_func

    def get(self, conf, context=None, scope=None):
        """
        Get an instance from configuration optional and context
        :param conf: The configuration, must have a name in the configuration
        :param context: The additional context to provide, it will be merged into "conf" as a dict and provide
        to the "conf_func"
        :param scope: The scope of the name
        :raise KeyError: When we doesn't have "name" in conf, or it is not registered, or the name
        is conflicted when scope is not provided
        :return: A configurable object
        """
        name = conf["name"]

        if scope is None:
            types_ = self.flat_map[name]
            if len(types_) != 1:
                raise KeyError(name)
            type_ = types_[0]
        else:
            # Scope is provided
            type_ = self.map[(scope, name)]

        merged_conf = {**(context or dict()), **conf}
        type_.conf_func(merged_conf)


_shared_conf_manager = ConfigManager()  # Global configuration manager


def register_conf(scope, name, conf_func=None):
    """
    A decorator to register a type to the shared configuration manager
    :param scope: The scope for registering
    :param name: The name for registering
    :param conf_func: An optional configuration function. If it is provided, it will be registered. If it is None, then
    the decorator will check whether the type has an "conf_func" attribute and register it. Or you can provide
    'conf_func="self"' to register the type itself as the configuration function
    :raise AttributeError: If conf_func is None and the registered type doesn't have "conf_func" attribute
    """
    def _register_conf(type_):
        actual_conf_func = type_ if conf_func == "self" else (conf_func or type_.conf_func)
        _shared_conf_manager.register_conf(scope, name, actual_conf_func)
        return type_
    return _register_conf
