import torch
import torchgeometry as tgm
import numpy as np

def normalize_quaternion(quaternion):
    """Normalize a quaternion."""
    norm = torch.norm(quaternion, p=2, dim=-1, keepdim=True)
    return quaternion / (norm + 1e-8)  # Adding a small epsilon to avoid division by zero

def average_rotations(rot_vec_list, first_element_weight_percentage=50.0):
    """
    Averages a list of rotation vectors with a specified weighting scheme. The first rotation vector 
    is assigned a weight according to the specified percentage of the total weight, emphasizing its significance.
    If the list contains only a single rotation vector, it is returned as is.

    Parameters:
    rot_vec_list (list of torch.Tensor): List of tensors of rotation vectors.
    first_element_weight_percentage (float): Percentage of the total weight for the first rotation vector.

    Returns:
    torch.Tensor: The averaged rotation vector.
    """
    # Handle single-element lists
    if len(rot_vec_list) == 1:
        return rot_vec_list[0]

    # Convert each list of rotation vectors to quaternions
    quats_list = [tgm.angle_axis_to_quaternion(rot_vec) for rot_vec in rot_vec_list]

    # Calculate the total weight and the weight of the first element
    total_elements_weight = len(quats_list) - 1
    first_element_weight = total_elements_weight * (first_element_weight_percentage / (100 - first_element_weight_percentage))
    weights = [first_element_weight] + [1] * (len(quats_list) - 1)
    
    # Apply weights
    weighted_quats = [weight * quat for weight, quat in zip(weights, quats_list)]
    
    # Stack all weighted quaternions into a single tensor for averaging
    quats_stack = torch.stack(weighted_quats, dim=0)
    
    # Compute the sum across the stacked dimension
    quat_sum = torch.sum(quats_stack, dim=0)
    
    # Normalize the sum to get the mean quaternion
    quat_mean_normalized = normalize_quaternion(quat_sum)
    
    # Convert the mean quaternion back to rotation vectors
    rot_vec_avg = tgm.quaternion_to_angle_axis(quat_mean_normalized)
    
    return rot_vec_avg

def calculate_weighted_averages_rot_2D(rot_2D_list, first_element_weight_percentage=70.0):
    """
    Applies average_rotations to each sublist of rotation vectors in a larger list, with a specified weight 
    for the first element in each sublist based on the given percentage of the total weight. If a sublist 
    contains only a single rotation vector, it is returned as the average without further calculation.

    Parameters:
    rot_2D_list (list of list of torch.Tensor): List containing sublists of rotation vectors.
    first_element_weight_percentage (float): The percentage of the total weight for the first element in each sublist.

    Returns:
    torch.Tensor: A tensor of the averaged rotation vectors for each sublist.
    """
    # Process each sublist
    averaged_rotations = []
    for rot_vec_list in rot_2D_list:
        if len(rot_vec_list) == 1:
            # If there's only one rotation vector, append it directly
            averaged_rotations.append(rot_vec_list[0])
        else:
            # Else, compute the weighted average
            averaged_rotations.append(average_rotations(rot_vec_list, first_element_weight_percentage))
    
    # Combine all averaged rotation vectors into a single tensor
    averaged_rotations_tensor = torch.stack(averaged_rotations, dim=0)
    
    return averaged_rotations_tensor

# Example usage:
# Assume rot_2D_list is a list of lists, where each sublist contains tensors of rotation vectors
# 70.0 represents the percentage of the total weight for the first element in each sublist
# averaged_rotations_tensor = calculate_weighted_averages_rot_2D(rot_2D_list, 70.0)





def calculate_weighted_averages_trans_2D(trans_2D_list, first_element_weight_percentage=75.0):
    """
    Computes the weighted averages of 2D translation parameters from a list of translation vector lists.
    The first translation vector in each sublist is assigned a preferential weight based on the specified percentage.
    In the case of a single-element sublist, the single translation vector is returned as is.

    Parameters:
    trans_2D_list (list of list of torch.Tensor): A list where each element is a sublist containing 2D translation vectors.
    first_element_weight_percentage (float): The percentage of the total weight to be assigned to the first element of each sublist.

    Returns:
    torch.Tensor: A tensor containing the weighted averages of the translation vectors from each sublist.
    """
    weighted_averages = []

    # Compute the weighted average for each sublist using the assigned weights
    for trans_list in trans_2D_list:
        # Check if there is only one element in the sublist
        if len(trans_list) == 1:
            # Return the single element as is
            weighted_averages.append(trans_list[0])
        else:
            # Calculate the total weight and the weight of the first element
            total_elements_weight = len(trans_list) - 1
            first_element_weight = total_elements_weight * (first_element_weight_percentage / (100 - first_element_weight_percentage))
            weights = [first_element_weight] + [1] * (len(trans_list) - 1)

            weighted_sum = sum(t * w for t, w in zip(trans_list, weights))
            weighted_average = weighted_sum / sum(weights)
            weighted_averages.append(weighted_average)
        
    # Combine the weighted averages into a single tensor
    weighted_averages_tensor = torch.stack(weighted_averages, dim=0)

    return weighted_averages_tensor



def calculate_weighted_averages_betas_2D(betas_2D_list, first_element_weight_percentage=50.0):
    """
    Calculates the weighted averages of 2D shape parameters from a list of shape parameter lists.
    This function assigns a preferential weight to the first shape parameter in each sublist based on a 
    specified percentage, which represents its relative importance in the weighted average calculation.
    If a sublist contains only a single shape parameter, it is returned as the weighted average.

    Parameters:
    betas_2D_list (list of list of torch.Tensor): A list where each element is a sublist containing 2D shape parameters.
    first_element_weight_percentage (float): The percentage of the total weight to be assigned to the first element of each sublist.

    Returns:
    torch.Tensor: A tensor containing the weighted averages of the shape parameters from each sublist.
    """
    weighted_averages = []

    # Calculate the weighted average for each sublist using the created weights
    for beta_list in betas_2D_list:
        # Handle single-element sublists by returning the element itself
        if len(beta_list) == 1:
            weighted_averages.append(beta_list[0])
        else:
            # Calculate the total weight and the weight of the first element
            total_elements_weight = len(beta_list) - 1  # Total weight of elements except the first
            first_element_weight = total_elements_weight * (first_element_weight_percentage / (100 - first_element_weight_percentage))
            weights = [first_element_weight] + [1] * (len(beta_list) - 1)

            weighted_sum = sum(b * w for b, w in zip(beta_list, weights))
            weighted_average = weighted_sum / sum(weights)
            weighted_averages.append(weighted_average)
        
    # Stack the weighted averages to form a tensor
    weighted_averages_tensor = torch.stack(weighted_averages, dim=0)

    return weighted_averages_tensor

# Example usage:
# betas_2D_list is your data list of lists
# 50.0 represents the percentage of the total weight for the first element
# weighted_betas = calculate_weighted_averages_betas_2D(betas_2D_list, 50.0)




def compute_cost_l2(pred_smpl_data, selected_track):
    """
    Input:
    pred_smpl_data: SMPL data in the world frame
    selected_track: SMPL data in the world frame (camera transformed)

    Output:
    Compute the L2 distance between each pair of 3D points
    """
    cost = np.sum(np.linalg.norm(pred_smpl_data - selected_track, axis=1))

    return cost



def normalize_smpl_features(features):
    # Assuming features is a 2D numpy array where each row is a feature vector
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    # Add a small epsilon to std to prevent division by zero
    epsilon = 1e-8
    normalized_features = (features - mean) / np.maximum(std, epsilon)
    return normalized_features

def normalize_appearance_embeddings(embeddings):
    # Normalize the appearance embeddings to have unit length
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized_embeddings = embeddings / (norms + 1e-8)  # Add epsilon to avoid division by zero
    return normalized_embeddings

def compute_cost_appearance(appearance1, appearance2):
    # Calculate the cosine distance between two appearance vectors
    # Compute the cosine similarity and then convert it to a distance measure
    cosine_similarity = np.dot(appearance1, appearance2)
    cosine_distance = 1 - cosine_similarity
    return cosine_distance

def compute_cost_appearance_euclidean(appearance1, appearance2):
    # Calculate the Euclidean distance between two appearance vectors
    distance = np.linalg.norm(appearance1 - appearance2)
    return distance