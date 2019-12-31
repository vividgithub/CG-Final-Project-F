import shutil
import re
import layers
from sgpn import PointNet, SGPNOutput
import tensorflow_core as tf
import math
from os import path, makedirs
import logger
from utils.kerasutil import ModelCallback
from utils.confutil import object_from_conf, register_conf
from SGPNLosses import SGPNloss

# A fake call to register
register_conf(name="adam", scope="optimizer", conf_func=lambda conf: tf.keras.optimizers.Adam(**conf))(None)
register_conf(name="sgd", scope="optimizer", conf_func=lambda conf: tf.keras.optimizers.SGD(**conf))(None)

register_conf(name="exponential_decay", scope="learning_rate",
              conf_func=lambda conf: tf.keras.optimizers.schedules.ExponentialDecay(**conf))(None)

_MODE_RESUME = "resume"
_MODE_NEW = "new"
_MODE_RESUME_COPY = "resume-copy"


def layer_from_config(layer_conf, model_conf, data_conf):
    """
    Get the corresponding keras layer from configurations
    :param layer_conf: The layer configuration
    :param model_conf: The global model configuration, sometimes it is used to generate some
    special layer like "output-classification" and "output-segmentation" layer
    :param data_conf: The dataset configuration, for generating special layers
    :return: A keras layer
    """
    context = {"class_count": data_conf["class_count"]}
    return object_from_conf(layer_conf, scope="layer", context=context)


def optimizer_from_config(learning_rate, optimizer_conf):
    """
    Get the optimizer from configuration
    :param learning_rate: The learning rate, might be a scalar or a learning rate schedule
    :param optimizer_conf: The optimizer configuration
    :return: An corresponding optimizer
    """
    context = {"learning_rate": learning_rate}
    return object_from_conf(optimizer_conf, scope="optimizer", context=context)


def learning_rate_from_config(learning_rate_conf):
    """
    Get the learning rate scheduler based on configuration
    :param learning_rate_conf: The learning rate configuration
    :return: A learning rate scheduler
    """
    return object_from_conf(learning_rate_conf, scope="learning_rate")


class GraphNode:
    """A class for representing a computation node in the graph. Use for generating"""
    def __init__(self, deps=None):
        """
        Initialization of a graph node
        :param deps: The dependencies graph node of this node. It make sure that all the dependencies node
        has been evaluated before the evaluation of current node
        """
        self.deps = deps or []
        self._value = None

    def add_dependency(self, dep):
        """
        Add a dependency
        :param dep: The dependency to add
        """
        self.deps.append(dep)

    def value(self):
        """
        Get the evaluation value of this node
        :return: The evaluation value
        """
        if self._value is not None:
            return self._value

        # Compute the dependencies
        dep_values = [dep.value() for dep in self.deps]
        self._value = self._compute(dep_values)
        return self._value

    def _compute(self, dep_values):
        """
        It determines how the value should be computed. Subclass should override this method
        :param dep_values: The value of the dependencies
        :return: The evaluation value
        """
        assert False, "The \"compute\" method of the GraphNode should not be called directly"
        return 0


class InputGraphNode(GraphNode):
    """Representing the node of the graph input for the network"""
    def __init__(self, inputs):
        super(InputGraphNode, self).__init__([])  # No dependency
        self._inputs = inputs

    def _compute(self, dep_values):
        return self._inputs


class IntermediateLayerGraphNode(GraphNode):
    """The intermediate computation node. It is usually a node wrapper for a keras layer"""
    def __init__(self, layer, deps=None):
        """
        Initialization
        :param layer: The keras computation layer, used to compute the evaluation value from the
        dependencies
        :param deps: The dependencies
        """
        super(IntermediateLayerGraphNode, self).__init__(deps)
        self._layer = layer

    def _compute(self, dep_values):
        logger.log(f"Computing node for layer {self._layer}")
        if len(dep_values) == 0:
            # Case 1, zero dependency, this should not occur
            assert False, f"No dependency for computing the layer {self._layer}, consider deleting it"
        elif len(dep_values) == 1:
            # Case 2, single dependency, call it directly
            return self._layer(dep_values[0])
        else:
            num_outputs = [len(dep_value) if isinstance(dep_value, (list, tuple)) else 0 for dep_value in dep_values]
            num_output = num_outputs[0]
            assert all([x == num_outputs for x in num_outputs]), \
                f"Cannot merge the dependencies since they have different number of outputs, num_outputs={num_outputs}"

            if num_output == 0:
                # Case 3, every dependencies generate only a single value, just concat them normally
                concat_value = tf.keras.layers.concatenate(dep_values, axis=-1, name="Concat")
            else:
                # Case 4, every dependencies generate multiple values. We need to concat them one by one
                # The values have been flattened before send into the layer
                concat_value = tf.keras.layers.Lambda(
                    lambda values: tuple([tf.concat(values[i::num_output], axis=-1) for i in range(num_output)]),
                    name="Concat"
                )(dep_values)

            return self._layer(concat_value)


class OutputGraphNode(IntermediateLayerGraphNode):
    """The final output graph node representation of the network"""
    def __init__(self, deps=None):
        super(OutputGraphNode, self).__init__(layer=tf.keras.layers.Lambda(tf.identity, name="Output"), deps=deps)


def net_from_config(model_conf, data_conf):
    """
    Generate a keras network from configuration dict
    :param model_conf: The global model configuration dictionary
    :param data_conf: The configuration of the dataset, it might use to initialize some layer like
    "output-classification"
    :param train_dataset: The train dataset, used to add input layer based on shape
    :return: A keras net
    """
    # Get network conf
    net_conf = model_conf["net"]

    # Input layer
    transform_confs = model_conf["dataset"].get("train_transforms", [])
    # Get the shape of the dataset, first check whether we have clip-feature layer in the dataset, if not, we
    # use the feature size in the dataset configuration
    feature_size = None
    for transform_conf in transform_confs[::-1]:
        if type(transform_conf) is dict and transform_conf.get("name") == "clip-feature":
            feature_size = transform_conf["c"]
            logger.log("Get feature_size={} from model configuration".format(feature_size))
    if feature_size is None:
        feature_size = data_conf.get("feature_size")
        logger.log("Get feature_size={} from dataset configuration".format(feature_size))
    assert feature_size is not None, "Cannot determine the feature_size"
    # Get the point size, if possible
    point_count = data_conf.get("point_count")
    for transform_conf in transform_confs[::-1]:
        if type(transform_conf) is dict and transform_conf.get("name") == "sampling":
            point_count = None
            logger.log("Ignore point_count since we have transform sampling from dataset")
    # input_layer = tf.keras.layers.InputLayer(input_shape=(point_count, feature_size))

    # Extend feature layer
    if "extend_feature" in net_conf:
        logger.log("\"extend_feature\" is deprecated, use \"input-feature-extend\" layer instead", color="yellow")

    #changed lines
    input1 = tf.keras.Input(shape=(point_count, feature_size))
    label1 = tf.keras.Input(shape=(point_count))
    label2 = tf.keras.Input(shape=(point_count))
    inputs = [input1, label1, label2]
    if net_conf["structure"] == "sequence":
        x = inputs  # Input layer

        '''
        shabidaimanimasile
        for layer_conf in net_conf["layers"]:
            logger.log(f"In constructing: {layer_conf}")
            layer = layer_from_config(layer_conf, model_conf, data_conf)
            logger.log(f"Input={x}")
            x = layer(x)
            logger.log(f"Output={x}")
        '''
        pointNetOutput = PointNet(x)
        outputs = SGPNOutput(pointNetOutput)
        #changed line
        return tf.keras.Model(inputs=inputs, outputs=outputs), inputs, outputs
    
    elif net_conf["structure"] == "graph":
        layer_confs = net_conf["layers"]
        graph_conf = net_conf["graph"]

        # Generate all the intermediate nodes and use label to map them
        nodes = []
        name_to_nodes = dict()
        for conf in layer_confs:
            node_name = conf.get("label", None)  # Use label to denote the layer
            node = IntermediateLayerGraphNode(layer_from_config(conf, model_conf, data_conf))
            nodes.append(node)
            if node_name is not None:
                assert node_name not in name_to_nodes, f"Layer name \"{node_name}\" conflict, check your labels"
                name_to_nodes[node_name] = node

        # Get the input node and output node
        input_node = InputGraphNode(inputs=inputs)
        output_node = OutputGraphNode()
        assert "input" not in name_to_nodes and "output" not in name_to_nodes, \
            f"Cannot name label of a layer to \"input\" or \"output\", check your layer labels"
        name_to_nodes["input"] = input_node
        name_to_nodes["output"] = output_node

        # Get the graph
        link_pattern = re.compile(r"[-]+>")  # ->, -->, --->, etc
        for s in graph_conf:
            logger.log(f"Analysing link \"{s}\"")
            n = re.split(link_pattern, s)  # Split
            n = [x.strip() for x in n]  # Remove space
            for i, node_name in enumerate(n):
                # Convert node_name(or index) to a node
                try:
                    node_index = int(node_name)
                    node = nodes[node_index]
                except ValueError:
                    assert node_name in name_to_nodes, f"Cannot find layer with label {node_name}"
                    node = name_to_nodes[node_name]
                n[i] = node
                if i > 0:
                    n[i].add_dependency(n[i - 1])

        for node in nodes + [output_node]:
            if len(node.deps) == 0:
                if isinstance(node, IntermediateLayerGraphNode):
                    logger.log(f"The node {node._layer} is isolated, please check your graph "
                               f"definitions", color="yellow")
                else:
                    logger.log("Output node is not linked", color="yellow")

        # Generate the network
        return tf.keras.Model(inputs=inputs, outputs=output_node.value())
    else:
        assert False, "\"{}\" is currently not supported".format(net_conf["structure"])


class ModelRunner:
    """
    A class to run a specified model on a specified dataset
    """

    def __init__(self, model_conf, data_conf, name, save_root_dir, train_dataset, test_dataset, mode=None):
        """
        Initialize a model runner
        :param model_conf: The pyconf for model
        :param data_conf: The pyconf for dataset
        :param name: The name for model
        :param save_root_dir: The root for saving. Normally it is the root directory where all the models of a specified
        dataset should be saved. Like something "path/ModelNet40-2048". Note that it is not the "root directory of the
        model", such as "path/ModelNet40-2048/PointCNN-X3-L4".
        :param train_dataset: The dataset to train the model
        :param test_dataset: The dataset to test the model
        :param mode: The mode indicates the strategy of whether to reuse the previous training process and continue
        training. Currently we support 3 types of modes:
            1. "new" or None: Do not use the previous result and start from beginning.
            2. "resume": Reuse previous result
            3. "resume-copy": Reuse previous result but make an exact copy.
        Both the "resume" and "resume-copy" will try to find the last result with the same "name" in the "save_root_dir"
        and reuse it. "resume" mode will continue training in the previous directory while "resume-copy" will try to
        create a new one and maintain the original one untouched. Default is None.
        """
        self.model_conf = model_conf
        self.data_conf = data_conf
        self.name = name
        self.save_root_dir = save_root_dir
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        self._mode = mode or "new"
        assert self._mode in [_MODE_NEW, _MODE_RESUME, _MODE_RESUME_COPY], \
            "Unrecognized mode=\"{}\". Currently support \"new\", \"resume\" and \"resume-copy\""

    def train(self):
        control_conf = self.model_conf["control"]

        # Transform the dataset is the dataset is classification dataset and
        # the model_conf's last output layer is output-conditional-segmentation
        train_dataset = test_dataset = None
        if self.data_conf["task"] == "classification" and \
                self.model_conf["net"]["layers"][-1]["name"] == "output-conditional-segmentation":
            layer_conf = self.model_conf["net"]["layers"][-1]
            assert "output_size" in layer_conf, "The dataset is classification dataset " \
                                                "while the model configuration is segmentation. " \
                                                "Cannot find \"output_size\" to transform the " \
                                                "classification dataset to segmentation task"
            seg_output_size = layer_conf["output_size"]
            # Transform function convert the label with (B, 1) to (B, N) where N is the last layer's point output size
            transform_func = (lambda points, label: (points, tf.tile(label, (1, seg_output_size))))
            train_dataset = self.train_dataset.map(transform_func)
            test_dataset = self.test_dataset
            logger.log("Convert classification to segmentation task with output_size={}".format(seg_output_size))
        else:
            train_dataset, test_dataset = self.train_dataset, self.test_dataset

        # Get the suffix of the directory by iterating the root directory and check which suffix has not been
        # created
        suffix = 0

        # The lambda tries to get the save directory based on the suffix
        def save_dir(suffix_=None):
            suffix_ = suffix_ if suffix_ is not None else suffix
            return path.join(self.save_root_dir, self.name + ("-" + str(suffix_) if suffix_ > 0 else ""))

        # Find the last one that the name has not been occupied
        while path.exists(save_dir()):
            suffix += 1

        # Check mode and create directory
        if self._mode == _MODE_NEW or suffix == 0:
            # We will enter here if the mode is "new" or we cannot find the previous model (suffix == 0)
            if self._mode != _MODE_NEW:
                logger.log("Unable to find the model with name \"{}\" to resume. Try to create new one", color="yellow")
            makedirs(save_dir(), exist_ok=False)
        elif self._mode == _MODE_RESUME:
            # Since we reuse the last one, we decrease it by one and do not need to create directory
            suffix -= 1
        elif self._mode == _MODE_RESUME_COPY:
            # Copy the reused one to the new one
            shutil.copytree(save_dir(suffix - 1), save_dir())
        logger.log("Save in directory: \"{}\"".format(save_dir()), color="blue")

        # Try get the infos and previous train step from the info.txt
        infos = dict()
        infos_file_path = path.join(save_dir(), "info.txt")
        if path.exists(infos_file_path) and path.isfile(infos_file_path):
            with open(path.join(save_dir(), "info.txt")) as f:
                pattern = re.compile(r"(\S+)[\s]?=[\s]*(\S+)")
                for line in f:
                    m = re.match(pattern, line.strip())
                    if m:
                        infos[m.group(1)] = eval(m.group(2))
            logger.log("Info loads, info: {}".format(logger.format(infos)), color="blue")
        else:
            logger.log("Do not find info, maybe it is a newly created model", color="blue")

        # Get the step offset
        # Because we store the "have trained" step, so it needs to increase by 1
        step_offset = infos.get("step", -1) + 1
        logger.log("Set step offset to {}".format(step_offset), color="blue")

        # Get the network
        logger.log("Creating network, train_dataset={}, test_dataset={}".format(self.train_dataset, self.test_dataset))
        #changed line
        net, inputs, outputs = net_from_config(self.model_conf, self.data_conf)

        # Get the learning_rate and optimizer
        logger.log("Creating learning rate schedule")
        lr_schedule = learning_rate_from_config(control_conf["learning_rate"])
        logger.log("Creating optimizer")
        optimizer = optimizer_from_config(lr_schedule, control_conf["optimizer"])

        # Get the loss
        #loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, name="Loss")
        loss = SGPNloss

        # Get the metrics
        # We add a logits loss in the metrics since the total loss will have regularization term
        metrics = [
            tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.SparseCategoricalCrossentropy(from_logits=True, name="logits_loss")
        ]

        # Get the batch size
        batch_size = control_conf["batch_size"]

        # Get the total step for training
        if "train_epoch" in control_conf:
            train_step = int(math.ceil(control_conf["train_epoch"] * self.data_conf["train"]["size"] / batch_size))
        elif "train_step" in control_conf:
            train_step = control_conf["train_step"]
        else:
            assert False, "Do not set the \"train_step\" or \"train_epoch\" in model configuraiton"

        # Get the validation step
        validation_step = control_conf.get("validation_step", None)
        tensorboard_sync_step = control_conf.get("tensorboard_sync_step", None) or validation_step or 100

        logger.log("Training conf: batch_size={}, train_step={}, validation_step={}, "
                   "tensorboard_sync_step={}".format(batch_size, train_step, validation_step, tensorboard_sync_step))

        # Get the callback
        # Initialize the tensorboard callback, and set the step_offset to make the tensorboard
        # output the correct step
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=save_dir(), update_freq=tensorboard_sync_step)
        if hasattr(tensorboard_callback, "_total_batches_seen"):
            tensorboard_callback._total_batches_seen = step_offset
        else:
            logger.log("Unable to set the step offset to the tensorboard, the scalar output may be a messy",
                       color="yellow")

        model_callback = ModelCallback(train_step, validation_step, train_dataset, test_dataset,
                                       batch_size, save_dir(), infos=infos, step_offset=step_offset)

        logger.log("Compile network, loss={}, metrics={}".format(loss, metrics))
        #changed line
        net.add_loss(loss([inputs[1], inputs[2]], outputs))
        net.trainable = True
        net.compile(optimizer, metrics=metrics)

        logger.log("Summary of the network:")
        net.summary(line_length=240, print_fn=lambda x: logger.log(x, prefix=False))

        logger.log("Begin training")
        net.fit(
            train_dataset,
            verbose=0,
            steps_per_epoch=train_step,
            callbacks=[tensorboard_callback, model_callback],
            shuffle=False  # We do the shuffle ourself
        )
        net.save_weights("sgpnmodel/sgpn.tf")