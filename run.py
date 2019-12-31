import argparse
from argparse import FileType as File
from logger import log
from utils import ioutil
from os.path import expanduser, abspath, isdir, exists
from os.path import join as join
from os import getcwd as cwd
from os import listdir, sep, makedirs

from utils.datasetutil import load_dataset
from utils.modelutil import ModelRunner

import tensorflow as tf
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.9)
config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
sess = tf.compat.v1.Session(config = config)

CLI_DESCRIPTION = """
Run a model based on provided model configuration file
"""

DATA_OPTIONS_DESCRIPTION = """
The data directory, should be the root of all the model data. 
For example, suppose the directory provided is "/home/local/Data", 
then it should contain the dataset data as its sub directory such as 
"/home/local/Data/ModelNet40-2048".
"""

SAVE_OPTIONS_DESCRIPTION = """
The saved directory, should be the root of all the model directory.
For example, suppose the directory provided is "/home/local/Models".
And current running dataset is ModelNet40-2048. Then the running model data will 
be saved in "/home/local/Models/ModelNet40-2048/${MODEL_NAME}".
"""

MODE_OPTIONS_DESCRIPTION = """
The mode to run the configuration. You could use "new" to create a 
new instance of the model. Or use "resume" to resume the previous task. 
You can also use "resume-copy" for resuming the last task but maintain the previous 
results (copy the latest execution result to a new directory).
"""


def parse_path(*paths):
    return abspath(expanduser(join(*paths)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=CLI_DESCRIPTION)
    parser.add_argument("model_config", type=str, default="conf/sgpn/sgpn.pyconf", help="The path for model configuration")
    parser.add_argument("-d", "--data", type=str, default="", help=DATA_OPTIONS_DESCRIPTION)
    parser.add_argument("-s", "--save", type=str, default="results", help=SAVE_OPTIONS_DESCRIPTION)
    parser.add_argument("-m", "--mode", type=str, default="new", help=MODE_OPTIONS_DESCRIPTION)
    args = parser.parse_args()

    model_conf_path = parse_path(args.model_config)
    root_data_dir = parse_path(args.data) if args.data else ""
    root_save_dir = parse_path(args.save) if args.save else ""

    mode = args.mode
    assert mode in ["new", "resume", "resume-copy"], f"Invalid mode \"{mode}\""

    assert model_conf_path.endswith(".pyconf"), \
        f"The model configuration \"{model_conf_path}\" is not a valid pyconf file"

    # Parse the path
    model_conf_path = parse_path(model_conf_path)
    log("Parsing model configuration")
    model_conf = ioutil.pyconf(model_conf_path)

    # Try to find the dataset name
    try:
        dataset_name = model_conf["dataset"]["name"]
    except KeyError:
        log("The name of the dataset cannot be found in the model configuration", color="red")
        exit(1)

    # Get the data directory
    log("Parse data configuration")
    if root_data_dir is None:
        log("The root of dataset directory is not provided, try finding...", color="yellow")

    # Scan up to 5 directory up
    scan_root_dirs = [parse_path(cwd(), *([".."] * i)) for i in range(5)] if not root_data_dir else [root_data_dir]
    # Scan each root directory to find the sub directory contains the name matches the dataset's name and has a
    # pyconf in it
    scan_sub_dirs = [parse_path(root_dir, x) for root_dir in scan_root_dirs for x in listdir(root_dir)
                     if x.lower() == "data"]
    scan_data_dirs = [parse_path(data_dir, x) for data_dir in scan_sub_dirs for x in listdir(data_dir)
                      if x.lower() == dataset_name.lower() and "conf.pyconf" in listdir(parse_path(data_dir, x))]

    try:
        data_dir = scan_data_dirs[0]
        root_data_dir = parse_path(data_dir, "..")
        log(f"✅️ {root_data_dir}")
    except IndexError:
        log(f"❌ Cannot find the data directory")
        exit(1)

    # Get the model save directory, if not provided, place it with name "Models" side by side the data directory
    log("Get the model saving directory")
    if not root_save_dir:
        root_save_dir = parse_path(root_data_dir, "..", "Models")
        log(f"The model saving root directory is not provided, saved in \"{root_save_dir}\"", color="yellow")
        if exists(root_save_dir) and not isdir(root_save_dir):
            log(f"Path \"{root_save_dir}\" is not a directory", color="red")
            exit(1)

    # Create the model save directory if needed
    save_dir = parse_path(root_save_dir, dataset_name)
    makedirs(save_dir, exist_ok=True)

    # Get the data conf
    train_dataset, test_dataset, data_conf = load_dataset(data_dir, model_conf)
    
    #print some of the dataset
    #fileout = open("tmp.txt","a")
#    count = 0
#    for i in train_dataset:
#        print(i)
#        count = count + 1
#        if count == 10: break

    # Initialize the ModelRunner
    model_name = model_conf_path.split(sep)[-1]  # Get the last component of the path
    model_name = model_name[:model_name.rfind(".")]  # Remove the .pyconf
    model_name = "".join([x.capitalize() for x in model_name.split("_")])  # Move lower case to camel case

    log(f"Model name: {model_name}")
    model_runner = ModelRunner(model_conf, data_conf, model_name, save_dir, train_dataset, test_dataset, mode=mode)

    log("Running model")
    model_runner.train()