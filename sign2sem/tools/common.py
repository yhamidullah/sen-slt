import gzip
import pickle

import yaml


def get_iter_gzip(sgn_path):
    """
    load feature pickle file
    :param sgn_path:
    :return:
    """
    f = gzip.open(sgn_path, "rb")
    folders = pickle.load(f)
    return folders


def load_config(path):
    """ load yaml config file
    :param path:
    :return:
    """
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg
