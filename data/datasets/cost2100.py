import os
import numpy as np
from PIL import Image, ImageChops
import scipy.io as sio
import random
import tensorflow as tf

def get_cost_2100_dataset(data_path, train=True, seed=1234, scenario="out"):

    data_path = os.path.join(data_path, "COST2100_outdoor/")

    dir_train = os.path.join(data_path, f"DATA_Htrain{scenario}.mat")
    dir_val = os.path.join(data_path, f"DATA_Hval{scenario}.mat")
    dir_test = os.path.join(data_path, f"DATA_Htest{scenario}.mat")
    dir_raw = os.path.join(data_path, f"DATA_HtestF{scenario}_all.mat")
    channel, nt, nc, nc_expand = 2, 32, 32, 125

    # Training data loading
    data_train = sio.loadmat(dir_train)['HT']
    data_train = np.array(data_train, dtype=np.float32).reshape(data_train.shape[0], channel, nt, nc)
    data_train = np.repeat(data_train[..., np.newaxis], 3, axis=-1)
    train_dataset = tf.data.Dataset.from_tensor_slices(data_train)

    # # Validation data loading
    # data_val = sio.loadmat(dir_val)['HT']
    # data_val = np.array(data_val, dtype=np.float32).reshape(data_val.shape[0], channel, nt, nc)
    # val_dataset = tf.data.Dataset.from_tensor_slices(data_val)

    data_test = sio.loadmat(dir_test)['HT']
    data_test = np.array(data_test, dtype=np.float32).reshape(data_test.shape[0], channel, nt, nc)
    data_test = np.repeat(data_test[..., np.newaxis], 3, axis=-1)
    test_dataset = tf.data.Dataset.from_tensor_slices(data_test)

    # raw_test = sio.loadmat(dir_raw)['HF_all']
    # real = np.real(raw_test).astype(np.float32)
    # imag = np.real(raw_test).astype(np.float32)
    # raw_test = np.concatenate((real.reshape(raw_test.shape[0], nt, nc_expand,1),
    #                            imag.reshape(raw_test.shape[0], nt, nc_expand,1)), axis=3)


    return train_dataset, test_dataset, None

