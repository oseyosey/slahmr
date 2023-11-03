import os
import re
import numpy as np
import pickle

def get_highest_motion_data(directory_path):
    """
    Given a directory path, find the npz file that ends with "_world_results.npz"
    and has the highest number preceding it.
    
    Parameters:
    - directory_path: Path to the directory to search in.
    
    Returns:
    - Filename with the highest number that ends with "_world_results.npz"
    """
    # List all files in the directory
    files = os.listdir(directory_path)

    # Filter files that end with "_world_results.npz"
    npz_files = [f for f in files if f.endswith("_world_results.npz")]

    # Extract the number from each filename and get the file with the highest number
    highest_number_file = max(npz_files, key=lambda f: int(re.search(r'(\d+)_world_results.npz', f).group(1)))

    highest_number_file_path = os.path.join(directory_path, highest_number_file)

    # Load the .npz file
    with np.load(highest_number_file_path) as data:
        # List all the arrays available in the file
        print(data.files)
        data_dict_slahmr_world = {key: data[key] for key in data.files}


    return data_dict_slahmr_world


def get_4D_human_data(keypoints_2d_path):
    keypoints_2d_path_data = f"{keypoints_2d_path}/complete_track_data.pkl" 
    # Load the lists from a pickle file
    with open(keypoints_2d_path_data, 'rb') as f:
        track_data = pickle.load(f)
    
    return track_data
        

