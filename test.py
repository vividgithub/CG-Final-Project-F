import argparse
from argparse import FileType as File
from logger import log
from utils import ioutil
from os.path import expanduser, abspath, isdir, exists
from os.path import join as join
from os import getcwd as cwd
from os import listdir, sep, makedirs
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from utils.datasetutil import load_dataset
from utils.modelutil import ModelRunner