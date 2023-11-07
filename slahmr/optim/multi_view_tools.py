
import torch
import torchgeometry as tgm

def normalize_quaternion(quaternion):
    """Normalize a quaternion."""
    norm = torch.norm(quaternion, p=2, dim=-1, keepdim=True)
    return quaternion / (norm + 1e-8)  # Adding a small epsilon to avoid division by zero

def average_rotations(rot_vec_list, weights=None):
    """
    Averages a list of rotation vectors with optional weighting.

    Parameters:
    rot_vec_list (list of torch.Tensor): List of tensors of rotation vectors.
    weights (list of float): Optional list of weights for each rotation vector.

    Returns:
    torch.Tensor: The averaged rotation vector.
    """
    # Convert each list of rotation vectors to quaternions
    quats_list = [tgm.angle_axis_to_quaternion(rot_vec) for rot_vec in rot_vec_list]
    
    # If no weights are provided, default to 1 for each quaternion, with the first having a weight of 2
    if weights is None:
        weights = [2] + [1] * (len(quats_list) - 1)
    
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

def calculate_weighted_averages_rot_2D(rot_2D_list):
    """
    Applies average_rotations to each sublist of rotation vectors in a larger list.

    Parameters:
    rot_2D_list (list of list of torch.Tensor): List containing sublists of rotation vectors.

    Returns:
    list of torch.Tensor: List of the averaged rotation vectors for each sublist.
    """
    averaged_rotations = [average_rotations(rot_vec_list) for rot_vec_list in rot_2D_list]
    
    averaged_rotations_tensor = torch.stack(averaged_rotations, dim=0)
    
    return averaged_rotations_tensor

# Example usage:
# Assume list_of_rot_lists is a list of lists, where each sublist contains tensors of rotation vectors
# list_of_rot_lists = [[tensor1, tensor2, ..., tensorN], [tensor1, tensor2, ..., tensorN], ...]
# averaged_list = average_all_rotations(list_of_rot_vec_lists)



def calculate_weighted_averages_trans_2D(trans_2D_list):
    """
    Computes the weighted averages of 2D translation parameters from a list of translation vector lists.
    Each sublist represents a series of 2D translations, and the function assigns a higher weight to the
    first translation vector in each sublist (weight of 2) compared to the others (weight of 1). This function
    is useful when the first translation vector is deemed more significant than the subsequent vectors in the list.

    Parameters:
    trans_2D_list (list of list of torch.Tensor): A list where each element is a sublist containing 2D translation vectors.

    Returns:
    torch.Tensor: A tensor containing the weighted averages of the translation vectors from each sublist.
    """
    weights_list = []

    # Assign weights with the first element having a weight of 2 and the rest 1 for each sublist
    for trans_list in trans_2D_list:
        weights = [2] + [1]*(len(trans_list)-1)
        weights_list.append(weights)

    weighted_averages = []

    # Compute the weighted average for each sublist using the assigned weights
    for trans_list, weights in zip(trans_2D_list, weights_list):
        weighted_average = sum(t * w for t, w in zip(trans_list, weights)) / sum(weights)
        weighted_averages.append(weighted_average)
        
    # Combine the weighted averages into a single tensor
    weighted_averages_tensor = torch.stack(weighted_averages, dim=0)

    return weighted_averages_tensor


def calculate_weighted_averages_betas_2D(betas_2D_list):
    """
    Calculates the weighted averages of 2D shape parameters from a list of shape parameter lists.
    Similar to calculate_weighted_averages_trans_2D, this function gives preferential weighting to
    the first shape parameter in each sublist (weight of 2) and standard weights (1) to the rest.
    This method emphasizes the influence of the first parameter when computing the average, which
    may represent a baseline or more significant shape feature.

    Parameters:
    betas_2D_list (list of list of torch.Tensor): A list where each element is a sublist containing 2D shape parameters.

    Returns:
    torch.Tensor: A tensor containing the weighted averages of the shape parameters from each sublist.
    """
    weights_list = []

    # Create a weights list with a heavier weight for the first element in each sublist
    for beta_list in betas_2D_list:
        weights = [2] + [1]*(len(beta_list)-1)
        weights_list.append(weights)

    weighted_averages = []

    # Calculate the weighted average for each sublist using the created weights
    for beta_list, weights in zip(betas_2D_list, weights_list):
        weighted_average = sum(b * w for b, w in zip(beta_list, weights)) / sum(weights)
        weighted_averages.append(weighted_average)
        
    # Stack the weighted averages to form a tensor
    weighted_averages_tensor = torch.stack(weighted_averages, dim=0)

    return weighted_averages_tensor
