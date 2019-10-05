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

    def register_conf(self, name, scope, conf_func):
        """
        Adds an entry to the configuration manager
        :param name: The name to register
        :param scope: The scope to search, should be in SCOPES
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

    def get(self, conf, scope=None, context=None):
        """
        Get an instance from configuration optional and context
        :param conf: The configuration, must have a name in the configuration
        :param scope: The scope of the name
        :param context: The additional context to provide, it will be merged into "conf" as a dict and provide
        to the "conf_func"
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


def register_conf(name, scope, conf_func=None, delete_name=True):
    """
    A decorator to register a type to the shared configuration manager
    :param name: The name for registering
    :param scope: The scope for registering
    :param conf_func: An optional configuration function. If it is provided, it will be registered. If it is None, then
    the decorator will check whether the type has an "conf_func" attribute and register it. Or you can provide
    'conf_func="self"' to register the type itself as the configuration function
    :param delete_name: Whether name should be deleted when trying to initializing an object with the configuration
    function
    :raise AttributeError: If conf_func is None and the registered type doesn't have "conf_func" attribute
    :raise AssertionError: If conf_func is "self" but the provided object is not callable
    """
    def _register_conf(type_):
        if conf_func == "self":
            assert callable(type_), f"{type_} is not callable"
            actual_conf_func = lambda conf: type_(**conf)
        else:
            actual_conf_func = conf_func or type_.conf_func

        # delete_name wrapper
        if delete_name:
            actual_conf_func = \
                lambda conf_with_name: actual_conf_func({k: v for k, v in conf_with_name.items() if k != "name"})

        _shared_conf_manager.register_conf(scope, name, actual_conf_func)
        return type_

    return _register_conf


def object_from_conf(conf, scope=None, context=None):
    """
    Get an instance from configuration
    :param conf: The configuration dictionary
    :param scope: The registered scope, maybe none to enable a global search if the name is unique
    :param context: Additional context to provide for initialization for an object
    :return: An instance
    """
    return _shared_conf_manager.get(conf, scope=scope, context=context)