import os
import tarfile
import numpy as np

# from .misc_data_util import transforms as trans
# from .misc_data_util.url_save import save
# from zipfile import ZipFile


def load_dataset(data_config):
    """
    Downloads and loads a variety of standard benchmark sequence datasets.
    Arguments:
        data_config (dict): dictionary containing data configuration arguments
    Returns:
        tuple of (train, val), each of which is a PyTorch dataset.
    """
    data_path = data_config["data_path"]  # path to data directory
    if data_path is not None:
        assert os.path.exists(data_path), "Data path {} not found.".format(data_path)

    # the name of the dataset to load
    dataset_name = data_config["dataset_name"]
    dataset_name = dataset_name.lower()  # cast dataset_name to lower case

    if dataset_name == "cost2100_outdoor":
        from .datasets import cost2100

        train, val, data_generator = cost2100.get_cost_2100_dataset(data_path=data_path)

    elif dataset_name == "cdl":
        train, val = None, None
        from data.datasets.cdl_c_multifrequency_multislot_dataset_loader import CDLChannelGenerator
        data_generator = CDLChannelGenerator(batch_size=data_config["batch_size"])
        
    elif dataset_name == "jscc":
        from .datasets import jscc
        train, val, data_generator = jscc.get_jscc_dataset(data_path=data_path)  

    elif dataset_name == "jscc_cost2100":
        from .datasets import jscc_cost2100
        train, val, data_generator = jscc_cost2100.get_jscc_cost2100_dataset(data_path=data_path)

    elif dataset_name == "jscc_complex":
        from .datasets import jscc_complex
        train, val, data_generator = jscc_complex.get_jscc_dataset(data_path=data_path)

    else:
        raise Exception("Dataset name not found.")

    return train, val, data_generator
