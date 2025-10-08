import numpy as np

def modify_uplink_variance(filepath, output_filepath):
    # Load the .npz file
    data = np.load(filepath)
    modified_data = {}
    
    # Iterate over each variable
    for key in data.files:
        array = data[key]
        variance_before = np.var(array)
        min_value = np.min(array)
        max_value = np.max(array)
        mean_value = np.mean(array)
        dtype = array.dtype  # Preserve original data type
        
        # Modify variance for variables containing "uplink"
        if "uplink" in key:
            std = np.sqrt(0.5)
            array = (array - mean_value) / np.std(array) * std + mean_value  # Adjust to have variance 0.5
            variance_after = np.var(array)
            print(f"Variable: {key}, Dimensions: {array.shape}, Variance Before: {variance_before}, Variance After: {variance_after}, Min: {min_value}, Max: {max_value}, Mean: {mean_value}")
        else:
            print(f"Variable: {key}, Dimensions: {array.shape}, Variance: {variance_before}, Min: {min_value}, Max: {max_value}, Mean: {mean_value}")
        array = array.astype(np.float32)
        modified_data[key] = array
    
    # Save the modified .npz file
    np.savez(output_filepath, **modified_data)
    print(f"Modified file saved at: {output_filepath}")

input_file = "/home/sa53869/csi_comp/datasets/jscc/jscc_sample_float32.npz"
output_file = "/home/sa53869/csi_comp/datasets/jscc/jscc_sample_float32_var.npz"  # Change to desired output path
modify_uplink_variance(input_file, output_file)
