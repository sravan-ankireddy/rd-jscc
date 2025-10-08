import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from global_config import PROJECTS_DIRECTORY

def load_jscc_dataset(data_path=None, mode="train", normalization=True):

    final_path = os.path.join(data_path, "jscc/QuadriGa_UMi_20_20_15.npz")

    data = np.load(final_path)

    print(f"Loading : {final_path}")
    
    x_freq_train_downlink = data['x_train_downlink'] - 0.5  # (100, 256, 32, 2)
    x_freq_val_downlink = data['x_val_downlink'] - 0.5
    x_freq_test_downlink = data['x_test_downlink'] - 0.5
    
    x_train_downlink = data['x_train_downlink'] - 0.5  # (100, 256, 32, 2)
    x_val_downlink = data['x_val_downlink'] - 0.5
    x_test_downlink = data['x_test_downlink'] - 0.5

    data_jscc = np.load(os.path.join(data_path, "jscc/jscc_float32_var.npz"))

    x_train_uplink = data_jscc['x_train_uplink']
    x_val_uplink = data_jscc['x_val_uplink']
    x_test_uplink = data_jscc['x_test_uplink']
    
    def preprocess_csi(x, normalization=True):
        # Combine real and imag into complex
        x_complex = x[..., 0] + 1j * x[..., 1]  # Shape: (N, 256, 32)

        # Transpose to (N, 32, 256)
        x_complex = np.transpose(x_complex, (0, 2, 1))

        # Apply 2D IFFT
        x_ifft = np.fft.ifft2(x_complex)

        # Crop to center 32x32
        x_cropped = x_ifft[:, :, :32]
        
        # compute the information lost because of cropping
        pad = np.zeros((x_cropped.shape[0], 32, x_ifft.shape[2]-x_cropped.shape[2]), dtype=np.complex64)
        
        x_cropped_pad = np.concatenate([x_cropped, pad], axis=2)
        
        x_reconstructed = np.fft.fft2(x_cropped_pad)
        
        # compute the error in nmse dB
        nmse = 10 * np.log10(np.mean(np.abs(x_complex - x_reconstructed)**2) / np.mean(np.abs(x_complex)**2))
        # print(f"NMSE: {nmse}")

        # Split into real and imag, and rearrange to (N, 2, 32, 32)
        x_processed = np.stack([x_cropped.real, x_cropped.imag], axis=1)  # (N, 2, 32, 32)
        
        # convert to float32
        x_processed = x_processed.astype(np.float32)

        if normalization:
            max_vals = np.max(np.abs(x_processed), axis=(1, 2, 3), keepdims=True)
                        
            x_processed /= (2 * max_vals)
            x_processed += 0.5
        return x_processed
    



    

    x_train_downlink = preprocess_csi(x_train_downlink)
    x_val_downlink = preprocess_csi(x_val_downlink)
    x_test_downlink = preprocess_csi(x_test_downlink)
    # breakpoint()
    # x_train_uplink = preprocess_csi(x_train_uplink)
    # x_val_uplink = preprocess_csi(x_val_uplink)
    # x_test_uplink = preprocess_csi(x_test_uplink)
    
    # get the first 32 subcarriers
    x_train_uplink = x_train_uplink[:, :32]
    x_val_uplink = x_val_uplink[:, :32]
    x_test_uplink = x_test_uplink[:, :32]

    # reaarange the dimensions
    x_train_uplink = np.transpose(x_train_uplink, (0, 3, 1, 2))
    x_val_uplink = np.transpose(x_val_uplink, (0, 3, 1, 2))
    x_test_uplink = np.transpose(x_test_uplink, (0, 3, 1, 2))    

    return x_train_downlink, x_val_downlink, x_test_downlink, x_train_uplink, x_val_uplink, x_test_uplink, x_freq_train_downlink, x_freq_val_downlink, x_freq_test_downlink

def visualize_samples(datasets, labels, datasets_spatial_frequency, indices=[0, 1, 2,3,4,5]):
    for idx in indices:
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        data_list = [datasets[idx], datasets_spatial_frequency[idx]]
        titles = ["CSI (Angular Delay Domain)", "CSI (Spatial Frequency Domain)"]

        for ax, data, title in zip(axes[:2], data_list, titles):
            img = ax.imshow(np.abs(data[0,:,:]+1j*data[1,:,:]), cmap='viridis')
            ax.set_title(f'{title}, idx:{idx}')
            fig.colorbar(img, ax=ax)

        plt.show()

def get_jscc_dataset(data_path=None, seed=42):
    # Load from the npy dataset and merge them all.
    train_dataset, _, test_dataset, sideinfo_train_dataset, _ , sideinfo_test_dataset, _, _, x_freq_test_downlink,   = load_jscc_dataset(data_path)
    x_freq_test_downlink = np.transpose(x_freq_test_downlink, (0, 3, 1, 2))

    train_dataset = np.stack([train_dataset, train_dataset, sideinfo_train_dataset], axis=-1)
    tf_train_dataset = tf.data.Dataset.from_tensor_slices(train_dataset)

    # test_dataset = np.stack([test_dataset, test_dataset, sideinfo_test_dataset], axis=-1)
    pad = np.zeros((x_freq_test_downlink.shape[0], 2, x_freq_test_downlink.shape[2]-32, 32), dtype=np.float32)
    test_dataset = np.concatenate([test_dataset, pad], axis=2)
    sideinfo_test_dataset = np.concatenate([sideinfo_test_dataset, pad], axis=2)
    # breakpoint()
    test_dataset = np.stack([test_dataset, test_dataset, sideinfo_test_dataset, x_freq_test_downlink], axis=-1)
    
    
    tf_test_dataset = tf.data.Dataset.from_tensor_slices(test_dataset)

    return tf_train_dataset, tf_test_dataset, None


if __name__ == "__main__":
    #tf_train_dataset, tf_test_dataset, _ = get_jscc_dataset(data_path=None, train=True, seed=1234, scenario="out")
    train_dataset, _, test_dataset, sideinfo_train_dataset, _ , sideinfo_test_dataset = load_jscc_dataset()
    visualize_samples(train_dataset, None, sideinfo_train_dataset)