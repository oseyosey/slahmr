import numpy as np
import matplotlib.pyplot as plt
import math
import torch

def plot_3d_points(list_of_vertices, ax=None, x_range=None, y_range=None, z_range=None, colors=None):
    if ax is None:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

    handles = []

    # If colors are not provided, generate random ones
    if colors is None:
        np.random.seed(45)
        colors = [np.random.rand(3,) for _ in list_of_vertices]

    for idx, vertices in enumerate(list_of_vertices):
        color = colors[idx] if colors else np.random.rand(3,)
        scatter = ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], marker='o', s=35, color=color, label=f"Subject {idx}")
        handles.append(scatter)
        centroid = vertices.mean(axis=0)
        ax.text(centroid[0], centroid[1], centroid[2], f"{idx}", color=color)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    if x_range:
        ax.set_xlim(x_range)
    if y_range:
        ax.set_ylim(y_range)
    if z_range:
        ax.set_zlim(z_range)

    legend = ax.legend(handles=handles, loc="upper right")
    for handle in legend.legendHandles:
        handle._sizes = [50]

    return ax


def plot_3d_joints(list_of_joints, ax=None, x_range=None, y_range=None, z_range=None, colors=None):
    if ax is None:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')

    handles = []

    # Default joint connections
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4), # arm
        (0, 5), (5, 6), (6, 7), (7, 8), # other arm
        (0, 9), (9, 10), (10, 11), (11, 12), # leg
        (0, 13), (13, 14), (14, 15), (15, 16), # other leg
        (0, 17), (17, 18), (18, 19), (19, 20), (20, 21) # spine and head
    ]

    # If colors are not provided, generate random ones
    if colors is None:
        np.random.seed(42)
        colors = [np.random.rand(3,) for _ in list_of_joints]

    for idx, joints in enumerate(list_of_joints):
        color = colors[idx] if colors else np.random.rand(3,)
        
        # Scatter joint coordinates
        scatter = ax.scatter(joints[:, 0], joints[:, 1], joints[:, 2], marker='o', s=50, color=color, label=f"Subject {idx}")
        handles.append(scatter)
        
        # Draw connections between joints
        for start, end in connections:
            ax.plot([joints[start, 0], joints[end, 0]], 
                    [joints[start, 1], joints[end, 1]], 
                    [joints[start, 2], joints[end, 2]], color=color)

        # Optionally: Label joints
        for j, coord in enumerate(joints):
            ax.text(coord[0], coord[1], coord[2], f"{j}", color=color)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    if x_range:
        ax.set_xlim(x_range)
    if y_range:
        ax.set_ylim(y_range)
    if z_range:
        ax.set_zlim(z_range)

    legend = ax.legend(handles=handles, loc="upper right")
    for handle in legend.legendHandles:
        handle._sizes = [50]

    return ax


def plot_multiple_graphs(data_list, central_title="", subtitles=[], matching=[]):
    num_graphs = len(data_list)
    # fig = plt.figure(figsize=(8 * num_graphs, 10))
    
    graphs_per_row = 3 if num_graphs % 3 == 0 else 2
    num_rows = math.ceil(num_graphs / graphs_per_row)
    fig = plt.figure(figsize=(20, 5 * num_rows))
    
        
    all_vertices = np.vstack([np.vstack(vertices_list) for vertices_list in data_list])
    x_range = (all_vertices[:, 0].min(), all_vertices[:, 0].max())
    y_range = (all_vertices[:, 1].min(), all_vertices[:, 1].max())
    z_range = (all_vertices[:, 2].min(), all_vertices[:, 2].max())
    
    
    # Step 1: Initialize color lists
    colors_for_all = [ [None]*len(data) for data in data_list ]
    
    # Step 2: Assign colors based on matching
    for match in matching:
        random_color = np.random.rand(3,)
        for idx, subject_index in enumerate(match):
            colors_for_all[idx][subject_index] = random_color
            
    # Step 3: Assign random colors to unmatched subjects
    for color_list in colors_for_all:
        for i in range(len(color_list)):
            if color_list[i] is None:
                color_list[i] = np.random.rand(3,)

    # Plot each graph:
    for idx, data in enumerate(data_list):
        ax = fig.add_subplot(1, num_graphs, idx+1, projection='3d')
        plot_3d_points(data, ax, colors=colors_for_all[idx], x_range=x_range, y_range=y_range, z_range=z_range)
        ax.set_title(subtitles[idx])

    fig.suptitle(central_title, fontsize=16)
    plt.show()

# Assuming plot_3d_points is defined as in your previous message



def plot_2d_joints(list_of_joints, ax=None, x_range=None, y_range=None, colors=None, canvas_height=None, canvas_width=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    
    # If canvas_height and canvas_width are specified, draw the background canvas
    if canvas_height and canvas_width:
        ax.set_xlim(0, canvas_width)
        ax.set_ylim(0, canvas_height)
    elif x_range:
        ax.set_xlim(x_range)
    if y_range:
        ax.set_ylim(y_range)

    handles = []

    # If colors are not provided, generate random ones
    if colors is None:
        np.random.seed(45)
        colors = [np.random.rand(3,) for _ in list_of_joints]

    for idx, joints in enumerate(list_of_joints):
        color = colors[idx] if colors else np.random.rand(3,)
        scatter = ax.scatter(joints[:, 0], joints[:, 1], marker='o', s=50, color=color, label=f"Subject {idx}")
        handles.append(scatter)
        for j, coord in enumerate(joints):
            ax.text(coord[0], coord[1], f"{j}", color=color)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    legend = ax.legend(handles=handles, loc="upper right")
    for handle in legend.legendHandles:
        handle._sizes = [50]

    return ax




def plot_combined(points_tensor: torch.Tensor, list_of_joints, canvas_width=None, canvas_height=None, colors=None):
    """
    Plots reprojected points and 2D joints on the same canvas.

    Args:
    - points_tensor (torch.Tensor): Reprojected points tensor of shape [batch_size, num_points, 2]
    - list_of_joints (list): List of 2D joint arrays, each of shape [num_joints, 2]
    - canvas_width (int): Width of the canvas (optional)
    - canvas_height (int): Height of the canvas (optional)
    - colors (list): List of RGB colors for each set of joints (optional)
    """
    
    fig, ax = plt.subplots(figsize=(10, 10))

    # If canvas_height and canvas_width are specified, draw the background canvas
    if canvas_width and canvas_height:
        ax.set_xlim(0, canvas_width)
        ax.set_ylim(0, canvas_height)

    # Plot reprojected points
    reprojected_np = points_tensor.numpy()
    for i in range(reprojected_np.shape[0]):
        ax.scatter(reprojected_np[i, :, 0], reprojected_np[i, :, 1], marker='x', s=60, label=f'Reprojected {i+1}')

    handles = []

    # If colors are not provided, generate random ones
    if colors is None:
        np.random.seed(45)
        colors = [np.random.rand(3,) for _ in list_of_joints]

    # Plot 2D joints
    for idx, joints in enumerate(list_of_joints):
        color = colors[idx] if colors else np.random.rand(3,)
        scatter = ax.scatter(joints[:, 0], joints[:, 1], marker='o', s=50, color=color, label=f"Subject {idx}")
        handles.append(scatter)
        for j, coord in enumerate(joints):
            ax.text(coord[0], coord[1], f"{j}", color=color)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    legend = ax.legend(loc="upper right")
    for handle in legend.legendHandles:
        handle._sizes = [50]

    plt.show()

# Example usage:
# plot_combined(reprojected_points, list_of_joints, canvas_width=640, canvas_height=480)




def plot_combined(reprojected_points, list_of_joints, image_width, image_height):
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot reprojected points
    for idx, points in enumerate(reprojected_points):
        color = 'C' + str(idx) # This will ensure unique colors for different subjects
        ax.scatter(points[:, 0], points[:, 1], marker='x', s=50, color=color, label=f'Reprojected {idx + 1}')
        for j, coord in enumerate(points):
            ax.text(coord[0], coord[1], f"{j}", color=color, ha='right') # Add the joint indices to reprojected points

    # Plot 2D joints
    if list_of_joints:
        colors = [plt.cm.rainbow(i) for i in np.linspace(0, 1, len(list_of_joints))] # Create distinct colors for each subject
        for idx, joints in enumerate(list_of_joints):
            color = colors[idx]
            ax.scatter(joints[:, 0], joints[:, 1], marker='o', s=50, color=color, label=f'Subject {idx}')
            for j, coord in enumerate(joints):
                ax.text(coord[0], coord[1], f"{j}", color=color, ha='left') # Add the joint indices to 2D joints

    ax.set_xlim(0, image_width)
    ax.set_ylim(0, image_height)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend(loc="upper right")
    plt.gca().invert_yaxis()  # Invert Y-axis for a more intuitive view (origin at the top-left as in images)
    plt.show()

# You can then use this function as before by passing the reprojected_points and list_of_joints


def plot_multiple_graphs(data_list, central_title="", subtitles=[], matching=[]):
    num_graphs = len(data_list)
    # fig = plt.figure(figsize=(8 * num_graphs, 10))
    
    graphs_per_row = 3 if num_graphs % 3 == 0 else 2
    num_rows = math.ceil(num_graphs / graphs_per_row)
    fig = plt.figure(figsize=(15, 6 * num_rows))
    
        
    all_vertices = np.vstack([np.vstack(vertices_list) for vertices_list in data_list])
    x_range = (all_vertices[:, 0].min(), all_vertices[:, 0].max())
    y_range = (all_vertices[:, 1].min(), all_vertices[:, 1].max())
    z_range = (all_vertices[:, 2].min(), all_vertices[:, 2].max())
    
    
    # Step 1: Initialize color lists
    colors_for_all = [ [None]*len(data) for data in data_list ]
    
    # Step 2: Assign colors based on matching
    for match in matching:
        random_color = np.random.rand(3,)
        for idx, subject_index in enumerate(match):
            colors_for_all[idx][subject_index] = random_color
            
    # Step 3: Assign random colors to unmatched subjects
    for color_list in colors_for_all:
        for i in range(len(color_list)):
            if color_list[i] is None:
                color_list[i] = np.random.rand(3,)

    # Plot each graph:
    for idx, data in enumerate(data_list):
        ax = fig.add_subplot(1, num_graphs, idx+1, projection='3d')
        plot_3d_points(data, ax, colors=colors_for_all[idx], x_range=x_range, y_range=y_range, z_range=z_range)
        ax.set_title(subtitles[idx])

    fig.suptitle(central_title, fontsize=16)
    plt.show()

# Assuming plot_3d_points is defined as in your previous message




import matplotlib.pyplot as plt
import matplotlib.patches as patches

def visualize_bboxes(image_path, bounding_boxes, save_path=None):
    """
    Visualize bounding boxes on an image using Matplotlib.

    Args:
    - image_path (str): Path to the image on which bounding boxes are to be drawn.
    - bounding_boxes (list[dict]): A list of bounding box information for each subject.
    - save_path (str, optional): Path to save the visualized image. If not provided, just shows the image.

    Returns:
    - None
    """
    
    # Load the image
    img = plt.imread(image_path)
    
    # Create a figure and axis
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    
    # Draw each bounding box
    for bbox in bounding_boxes:
        x_top_left = bbox['x_top_left']
        y_top_left = bbox['y_top_left']
        width = bbox['x_bottom_right'] - x_top_left
        height = bbox['y_bottom_right'] - y_top_left
        
        rect = patches.Rectangle((x_top_left, y_top_left), width, height, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    
    # Display the image
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def visualize_bboxes_tlwh(image_path, bounding_boxes, save_path=None):
    """
    Visualize bounding boxes on an image using Matplotlib.

    Args:
    - image_path (str): Path to the image on which bounding boxes are to be drawn.
    - bounding_boxes (list[dict]): A list of bounding box information for each subject.
    - save_path (str, optional): Path to save the visualized image. If not provided, just shows the image.

    Returns:
    - None
    """
    
    # Load the image
    img = plt.imread(image_path)
    
    # Create a figure and axis
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    
    # Draw each bounding box
    for bbox in bounding_boxes:
        x_top_left = bbox['x_top_left']
        y_top_left = bbox['y_top_left']
        width = bbox['width']
        height = bbox['height']
        
        rect = patches.Rectangle((x_top_left, y_top_left), width, height, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    
    # Display the image
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()