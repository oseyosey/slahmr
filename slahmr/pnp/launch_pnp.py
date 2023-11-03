import numpy as np
import cv2
import re
import torch
import pickle
import sys
import os

## Append system directory for SMPL functions
ROOT_DIR = os.path.abspath(f"{__file__}/../")
print("PROJ SRC for Perspective-N-Point Function: ", ROOT_DIR)
sys.path.append(ROOT_DIR)

from pnp.pnp_helpers import *
from util.loaders import (
    load_vposer,
    load_state,
    load_gmm,
    load_smpl_body_model,
    resolve_cfg_paths,
)
from body_model import SMPL_JOINTS, KEYPT_VERTS, smpl_to_openpose, run_smpl
from geometry.camera import perspective_projection


def run_pnp(cfg, keypoints_2d_path_mv, keypoints_3d_path, cv_match_path, device):
    """
    Solve for PnP to obtain Camera Pose
    # 3D keypoints path from SLAHMR:
    # 2D keypoints path from 4D-Human(PHALP): 
    # Cross view matching path: 
    # Obtain cross view matching: return a list of 
    """
    ## Load cross view matching
    with open(cv_match_path, "rb") as f:
        cross_view_matching = pickle.load(f)
    # frame = len(smpl_data_world['joints3d_op'][0])
    frame = len(cross_view_matching['cross_view_match'])



    ## Load 3D keypoints for SLAHMR (single vieww)
    keypoints_3d_motion_path = f"{keypoints_3d_path}/motion_chunks/" 
    data_dict_slahmr_world = get_highest_motion_data(keypoints_3d_motion_path)
    smpl_data_world = transform_smpl_paramas(data_dict_slahmr_world, device)
    joints_3d_data_all_frames = []


    for j in range(frame):
        joints_3d_data_per_frame = []
        for i in range(smpl_data_world['joints3d_op'].shape[0]):
            joints_3d_data_per_frame.append(smpl_data_world['joints3d_op'][i][j].cpu().detach().numpy())
        joints_3d_data_all_frames.append(joints_3d_data_per_frame)


    ## Load 2D keypoints (for multi-view)
    track_data_mv = []
    joints_2d_data_mv = []
    for num_view in range(1, cfg.data.multi_view_num):
        keypoints_2d_path = keypoints_2d_path_mv[num_view] 
        track_data = get_4D_human_data(keypoints_2d_path) # List of length B, each contain (T, 25, 3).
        track_data_mv.append(track_data)
        joints_2d_data_all_frames = [] # List of length T, [subject 1, 2, 3 at time 0; subject 1, 2, 3 at time 1; ... ]
        for j in range(frame):
            joints_2d_data_all_frames_subjects = []
            for i in range(len(track_data['joints2d'])):
                joints_2d_subject = track_data['joints2d'][i][j][..., :2]
                joints_2d_data_all_frames_subjects.append(joints_2d_subject)
            joints_2d_data_all_frames.append(joints_2d_data_all_frames_subjects)

        joints_2d_data_mv.append(joints_2d_data_all_frames)


    ## Obtain camera pose for each frame
    rt_pairs = []
    #append rt_pairs with (R=Eye(), T=zeros(). 
    rt_pairs.append((np.eye(3) ,np.zeros((3, 1)))) 

            
    fx = data_dict_slahmr_world['intrins'][0]
    fy = data_dict_slahmr_world['intrins'][1]
    cx = data_dict_slahmr_world['intrins'][2]
    cy = data_dict_slahmr_world['intrins'][3]
    camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
       
    ## Matching 3D Keypoints with 2D Keypoints
    for num_view in range(1, cfg.data.multi_view_num):
        
        points_3D_world = []
        points_2D_camera = []
        breakpoint()
        for t_ in range(frame):
            ## Obtain BBoox from 2D keypoitns at camera coordiante
            joints_2d_data = joints_2d_data_mv[num_view-1][t_] ## num_view - 1 because we didn't store view 0 (world camera)
            joints_2d_detected_camera = np.array(joints_2d_data)
            bbox_2d_detected_camera = joints_2d_to_bboxes(joints_2d_detected_camera)

            ## Obtain BBox from 3D keypoints at world coordinate
            joints_3d_data = joints_3d_data_all_frames[t_]
            joints_3d_world = np.array(joints_3d_data)
            bbox_2d_reprojected_world = joints_3d_to_bboxes(joints_3d_world, data_dict_slahmr_world)

            ## check for empty list
            if len(bbox_2d_detected_camera) != 0 and len(bbox_2d_reprojected_world) != 0 and len(cross_view_matching['cross_view_match'][t_]) !=0:
                match_pairs_per_frame = match_two_bbox_sets(bbox_2d_reprojected_world, bbox_2d_detected_camera, cross_view_matching['cross_view_match'], frame=t_, camera1=0, camera2=num_view)
                for pair in match_pairs_per_frame:
                    points_3D_world.append(joints_3d_data[pair[0]])
                    points_2D_camera.append(joints_2d_data[pair[1]])


        points_3D_world = np.vstack(points_3D_world)
        points_2D_camera = np.vstack(points_2D_camera)
        
        breakpoint()
        print(f"RUNNING RANSAC PnP ROBUST on Camera {num_view}")
        n = int(len(points_3D_world) / (25*5))

        ## Need additional parameter finetuning
        refined_pose = ransac_pnp_robust(points_3D_world, points_2D_camera, camera_matrix, n=n, k=1000, d=25 * (n/2), percentile=95, sample_ratio=0.8, verbose=False)
        rt_pairs.append(refined_pose)

    return rt_pairs



def transform_smpl_paramas(data_dict_slahmr_world, device):
    """
    Obtain SMPL results (containing 3D joints information) from SLAHMR results.
    """  
    ## Running "Solution: obtaining 3D joints" Section
    trans = torch.from_numpy(data_dict_slahmr_world['trans']).to(device) #subject 0, frame 0 
    root_orient = torch.from_numpy(data_dict_slahmr_world['root_orient']).to(device)
    body_pose = torch.from_numpy(data_dict_slahmr_world['pose_body']).to(device)
    betas = torch.from_numpy(data_dict_slahmr_world['betas']).to(device)

    SLAHMR_DIR = os.path.abspath(f"{__file__}/../../../")
    path_smpl = os.path.join(SLAHMR_DIR, "_DATA/body_models/smplh/neutral/model.npz")


    T = data_dict_slahmr_world['root_orient'].shape[1]
    B = data_dict_slahmr_world['root_orient'].shape[0]
    body_model, fit_gender = load_smpl_body_model(path_smpl, B * T, device=device)

    smpl_results = pred_smpl(trans, root_orient, body_pose, betas, body_model)

    return smpl_results



def pred_smpl(trans, root_orient, body_pose, betas, body_model):
    """
    Forward pass of the SMPL model and populates pred_data accordingly with
    joints3d, verts3d, points3d.

    trans : B x T x 3
    root_orient : B x T x 3
    body_pose : B x T x J*3
    betas : B x D
    """
    smpl2op_map = smpl_to_openpose(
            body_model.model_type,
            use_hands=False,
            use_face=False,
            use_face_contour=False,
            openpose_format="coco25",
        )

    smpl_out = run_smpl(body_model, trans, root_orient, body_pose, betas)
    joints3d, points3d = smpl_out["joints"], smpl_out["vertices"]

    # select desired joints and vertices
    joints3d_body = joints3d[:, :, : len(SMPL_JOINTS), :]
    joints3d_op = joints3d[:, :, smpl2op_map, :]
    # hacky way to get hip joints that align with ViTPose keypoints
    # this could be moved elsewhere in the future (and done properly)
    joints3d_op[:, :, [9, 12]] = (
        joints3d_op[:, :, [9, 12]]
        + 0.25 * (joints3d_op[:, :, [9, 12]] - joints3d_op[:, :, [12, 9]])
        + 0.5
        * (
            joints3d_op[:, :, [8]]
            - 0.5 * (joints3d_op[:, :, [9, 12]] + joints3d_op[:, :, [12, 9]])
        )
    )
    verts3d = points3d[:, :, KEYPT_VERTS, :]

    return {
        "points3d": points3d,  # all vertices
        "verts3d": verts3d,  # keypoint vertices
        "joints3d": joints3d_body,  # smpl joints
        "joints3d_op": joints3d_op,  # OP joints
        "faces": smpl_out["faces"],  # index array of faces
    }




def solve_PnP(points_3D, points_2D, camera_matrix, dist_coeffs=None):
    """
    Generic PnP solution using OpenCV
    Needs at least 6-point correspondences to perform PnP 
    points_3d has the np.shape = (Total Number of joints, 3)
    points_2d has the np.shape = (Total Number of joints, 2)
    """
    # Use solvePnP
    ret, rvec, tvec = cv2.solvePnP(points_3D, points_2D, camera_matrix, dist_coeffs)
    return ret, rvec, tvec



def filter_zero_points(points_2D, points_3D):
    """
    Removes points from points_2D and points_3D where points_2D contains a 0 in either x or y.

    Parameters:
    - points_2D: The 2D points to check.
    - points_3D: The corresponding 3D points.

    Returns:
    - filtered_points_2D, filtered_points_3D: The filtered 2D and 3D points.
    """

    # Find indices where neither x nor y in points_2D is 0
    valid_indices = np.logical_and(points_2D[:,0] != 0, points_2D[:,1] != 0)

    # Filter points_2D and points_3D using the valid indices
    filtered_points_2D = points_2D[valid_indices]
    filtered_points_3D = points_3D[valid_indices]

    return filtered_points_2D, filtered_points_3D



def ransac_pnp(points_3D, points_2D, camera_matrix, n=6, k=30, d=25*6, percentile=95):
    """
    points_3D: 3D Joints in world coordinate (25 * number_of_samples, 3)
    points_2D: 2D Joints in camera coordiante (25 * number_of_samples, 3)
    param n: Number of subjects to sample in each iteration
    param k: Number of iterations
    param d: Number of inliers required to accept a model
    RANSAC for PnP problem with subject-level random sampling and refined model estimation.
    """
    num_subjects = len(points_3D) // 25
    t = 0  # Placeholder; will be overwritten later
    
    # Initial pose estimation
    points_2D_filtered, points_3D_filtered = filter_zero_points(points_2D, points_3D)
    _, rvec_init, tvec_init = cv2.solvePnP(np.array(points_3D_filtered), np.array(points_2D_filtered), camera_matrix, None)
    projected_2D_init, _ = cv2.projectPoints(np.array(points_3D_filtered), rvec_init, tvec_init, camera_matrix, None)
    residuals = np.linalg.norm(points_2D_filtered - projected_2D_init.squeeze(), axis=1)
    t = np.percentile(residuals, percentile)
    print("Initial Threshold: ", t)
    
    best_model = None
    best_inliers = None
    best_error = float('inf')
    
    for iteration in range(k):
        # Randomly select subjects
        subject_idx = np.random.choice(num_subjects, n, replace=False)
        subject_3D = np.concatenate([points_3D[i*25:(i+1)*25] for i in subject_idx])
        subject_2D = np.concatenate([points_2D[i*25:(i+1)*25] for i in subject_idx])
        
        # Filter out zero points from subject's data
        subject_2D, subject_3D = filter_zero_points(subject_2D, subject_3D)
        
        # Assuming no lens distortion
        dist_coeffs = np.zeros((4,1), dtype=np.float32)
        
        # Estimate pose using the chosen subjects
        _, rvec, tvec = cv2.solvePnP(np.array(subject_3D), np.array(subject_2D), camera_matrix, dist_coeffs)
        
        # Calculate reprojected points and error
        projected_2D, _ = cv2.projectPoints(np.array(points_3D_filtered), rvec, tvec, camera_matrix, dist_coeffs)
        error = np.linalg.norm(points_2D_filtered - projected_2D.squeeze(), axis=1)
        inliers = error < t
        
        if np.sum(inliers) >= d:
            current_error = np.mean(error[inliers])
            if current_error < best_error:
                best_model = (rvec, tvec)
                best_inliers = inliers
                best_error = current_error
                print("Updated Best Error: ", current_error)
    
    if best_model is None:
        raise ValueError("RANSAC did not find a suitable fit.")

    # Model refinement using the inliers of the best model
    _, rvec_refined, tvec_refined = cv2.solvePnP(np.array(points_3D_filtered)[best_inliers], np.array(points_2D_filtered)[best_inliers], camera_matrix, None)
    refined_model = (rvec_refined, tvec_refined)
    
    return refined_model


def ransac_pnp_robust(points_3D, points_2D, camera_matrix, n=6, k=30, d=25*6, percentile=95, sample_ratio=0.8, verbose = False):
    """
    points_3D: 3D Joints in world coordinate (25 * number_of_samples, 3)
    points_2D: 2D Joints in camera coordiante (25 * number_of_samples, 3)
    param n: Number of subjects to sample in each iteration
    param k: Number of iterations
    param d: Number of inliers required to accept a model
    param sample_ratio: Fraction of points to sample in the second random sampling stage
    RANSAC for PnP problem with subject-level random sampling, zero point filtering, and refined model estimation.
    """
    num_subjects = len(points_3D) // 25
    t = 0

    # Assuming no lens distortion
    dist_coeffs = np.zeros((4,1), dtype=np.float32)
    
    # Initial pose estimation
    points_2D_filtered, points_3D_filtered = filter_zero_points(points_2D, points_3D)
    _, rvec_init, tvec_init = cv2.solvePnP(np.array(points_3D_filtered), np.array(points_2D_filtered), camera_matrix, dist_coeffs)
    projected_2D_init, _ = cv2.projectPoints(np.array(points_3D_filtered), rvec_init, tvec_init, camera_matrix, dist_coeffs)
    residuals = np.linalg.norm(points_2D_filtered - projected_2D_init.squeeze(), axis=1)
    t = np.percentile(residuals, percentile)

    if verbose:
        print("Initial Threshold: ", t)
        print("Init Rotation Vector: ", rvec_init)
        print("Init Translation Vector: ", tvec_init)

    best_model = None
    best_inliers = None
    best_error = float('inf')
    
    for iteration in range(k):
        # Randomly select subjects
        subject_idx = np.random.choice(num_subjects, n, replace=False)
        subject_3D = np.concatenate([points_3D[i*25:(i+1)*25] for i in subject_idx])
        subject_2D = np.concatenate([points_2D[i*25:(i+1)*25] for i in subject_idx])
        
        # Filter out zero points
        subject_2D_filtered, subject_3D_filtered = filter_zero_points(subject_2D, subject_3D)
        
        # Additional Random Sampling at the point level
        point_indices = np.random.choice(len(subject_3D_filtered), int(len(subject_3D_filtered)*sample_ratio), replace=False)
        subject_3D_filtered = subject_3D_filtered[point_indices]
        subject_2D_filtered = subject_2D_filtered[point_indices]
        
        # Estimate pose using the chosen points
        _, rvec, tvec = cv2.solvePnP(np.array(subject_3D_filtered), np.array(subject_2D_filtered), camera_matrix, dist_coeffs)
        
        # Calculate reprojected points and error
        projected_2D, _ = cv2.projectPoints(np.array(points_3D_filtered), rvec, tvec, camera_matrix, dist_coeffs)
        error = np.linalg.norm(points_2D_filtered - projected_2D.squeeze(), axis=1)
        inliers = error < t
        
        if np.sum(inliers) >= d:
            current_error = np.mean(error[inliers])
            if current_error < best_error:
                best_model = (rvec, tvec)
                best_inliers = inliers
                best_error = current_error
                if verbose:
                    print("Updated Best Error: ", best_error)
    
    if best_model is None:
        raise ValueError("RANSAC did not find a suitable fit.")

    # Model refinement using the inliers of the best model
    _, rvec_refined, tvec_refined = cv2.solvePnP(np.array(points_3D_filtered)[best_inliers], np.array(points_2D_filtered)[best_inliers], camera_matrix, None)
    refined_model = (rvec_refined, tvec_refined)
    
    return refined_model



def extract_bbox_info(image_path):
    # Use regular expression to extract bounding box coordinates, camera ID, and subject ID
    pattern = re.compile(r'Camera(\d+)_(\d+)_(\d+)_(\d+)_(\d+)_(\d+).jpg')
    match = pattern.search(image_path)
    if match:
        camera_id = match.group(1)
        subject_id = match.group(2)
        x_top_left = int(match.group(3))
        y_top_left = int(match.group(4))
        x_bottom_right = int(match.group(5))
        y_bottom_right = int(match.group(6))
        
        # Store the extracted information in a dictionary
        info = {
            'camera_id': camera_id,
            'subject_id': subject_id,
            'bbox': {
                'x_top_left': x_top_left,
                'y_top_left': y_top_left,
                'width': x_bottom_right,
                'height': y_bottom_right,
            }
        }
        return info
    else:
        print(f"Could not extract information from image path: {image_path}")
        return None



def prepare_perspective_projection_data(joints_3d_world, data_dict_slahmr_world):   
# Convert each NumPy array in the list to a PyTorch tensor
    points_3D_tensors = [torch.tensor(point) for point in joints_3d_world]

    # Now, convert the list of tensors to a single tensor
    joints_3d_world_tensors = torch.stack(points_3D_tensors, dim=0)

    bs = len(joints_3d_world)  # batch size

    focal_length = data_dict_slahmr_world['intrins'][0]
    # Create the focal_length_tensor
    focal_length_tensor = torch.full((bs, 2), focal_length)


    # rotation
    # translation
    rotation_list = []
    translation_list = []
    for i in range(len(data_dict_slahmr_world['cam_R'])):
        rotation_list.append(data_dict_slahmr_world['cam_R'][i][0])
        translation_list.append(data_dict_slahmr_world['cam_t'][i][0])

    # Convert the list of rotations into a tensor of shape (bs, 3, 3)
    rotation_tensor = torch.stack([torch.tensor(item) for item in rotation_list], dim=0)

    # Convert the list of translations into a tensor of shape (bs, 3)
    translation_tensor = torch.stack([torch.tensor(item) for item in translation_list], dim=0)


    # Create the camera_center_tensor
    cx = data_dict_slahmr_world['intrins'][2]
    cy = data_dict_slahmr_world['intrins'][3]
    camera_center_tensor = torch.tensor([[cx, cy]] * bs)

    return joints_3d_world_tensors, focal_length_tensor, camera_center_tensor, rotation_tensor, translation_tensor

def joints_3d_to_bboxes(joints_3d_world, data_dict_slahmr_world):
    """
    Convert 3D joint coordinates to bounding box information.
    1. 3D perspective Reprojection to 2D joints
    2. Extract Bounding Box from 2D joints

    Args:
    - reprojected_3d_joints (torch.Tensor): A tensor of shape (num_subjects, num_joints, 2) 
      containing 2D joint coordinates for each subject.

    Returns:
    - list[dict]: A list of bounding box information for each subject.
    """

    joints_3d_world_tensor, focal_length_tensor, camera_center_tensor, rotation_tensor, translation_tensor = prepare_perspective_projection_data(joints_3d_world, data_dict_slahmr_world)

    reprojected_3d_joints = perspective_projection(joints_3d_world_tensor, focal_length_tensor, camera_center_tensor, rotation = rotation_tensor, translation = translation_tensor)


    # Convert the tensor to a numpy array
    if torch.is_tensor(reprojected_3d_joints):
        reprojected_3d_joints = reprojected_3d_joints.numpy()
        
    # A list to store bounding box information for each subject
    bounding_boxes = []

    # Extract bounding box for each subject's set of 2D joints
    for subject_joints in reprojected_3d_joints:
        # Filter out joints with (0,0) coordinates
        valid_joints = subject_joints[np.all(subject_joints != [0,0], axis=1)]
        
        # Check if there are any valid joints left after filtering
        if valid_joints.size == 0:
            continue

        x_coords = valid_joints[:, 0]
        y_coords = valid_joints[:, 1]
        
        # Minimum and maximum values
        x_top_left = np.min(x_coords)
        y_top_left = np.min(y_coords)
        x_bottom_right = np.max(x_coords)
        y_bottom_right = np.max(y_coords)
        
        # Store the bounding box information
        bbox = {
            'x_top_left': x_top_left,
            'y_top_left': y_top_left,
            'x_bottom_right': x_bottom_right,
            'y_bottom_right': y_bottom_right,
        }
        bounding_boxes.append(bbox)
    
    return bounding_boxes



def joints_2d_to_bboxes(reprojected_points):
    """
    Convert 2D joint coordinates to bounding box information.

    Args:
    - reprojected_points (torch.Tensor): A tensor of shape (num_subjects, num_joints, 2) 
      containing 2D joint coordinates for each subject.

    Returns:
    - list[dict]: A list of bounding box information for each subject.
    """
    
    # Convert the tensor to a numpy array
    if torch.is_tensor(reprojected_points):
        reprojected_points = reprojected_points.numpy()
        
    # A list to store bounding box information for each subject
    bounding_boxes = []

    # Extract bounding box for each subject's set of 2D joints
    for subject_joints in reprojected_points:
        # Filter out joints with (0,0) coordinates
        valid_joints = subject_joints[np.all(subject_joints != [0,0], axis=1)]
        
        # Check if there are any valid joints left after filtering
        if valid_joints.size == 0:
            continue

        x_coords = valid_joints[:, 0]
        y_coords = valid_joints[:, 1]
        
        # Minimum and maximum values
        x_top_left = np.min(x_coords)
        y_top_left = np.min(y_coords)
        x_bottom_right = np.max(x_coords)
        y_bottom_right = np.max(y_coords)
        
        # Store the bounding box information
        bbox = {
            'x_top_left': x_top_left,
            'y_top_left': y_top_left,
            'x_bottom_right': x_bottom_right,
            'y_bottom_right': y_bottom_right,
        }
        bounding_boxes.append(bbox)
    
    return bounding_boxes


def compute_iou(bbox1, bbox2):
    """
    Compute the Intersection over Union (IoU) between two bounding boxes.
    """
    # Convert bbox1 to (x_tl, y_tl, x_br, y_br) format if it's in (x_tl, y_tl, width, height) format
    if 'width' in bbox1:
        bbox1['x_bottom_right'] = bbox1['x_top_left'] + bbox1['width']
        bbox1['y_bottom_right'] = bbox1['y_top_left'] + bbox1['height']

    # Convert bbox2 to (x_tl, y_tl, x_br, y_br) format if it's in (x_tl, y_tl, width, height) format
    if 'width' in bbox2:
        bbox2['x_bottom_right'] = bbox2['x_top_left'] + bbox2['width']
        bbox2['y_bottom_right'] = bbox2['y_top_left'] + bbox2['height']

    xA = max(bbox1['x_top_left'], bbox2['x_top_left'])
    yA = max(bbox1['y_top_left'], bbox2['y_top_left'])
    xB = min(bbox1['x_bottom_right'], bbox2['x_bottom_right'])
    yB = min(bbox1['y_bottom_right'], bbox2['y_bottom_right'])
    
    interArea = max(0, xB - xA) * max(0, yB - yA)
    box1Area = (bbox1['x_bottom_right'] - bbox1['x_top_left']) * (bbox1['y_bottom_right'] - bbox1['y_top_left'])
    box2Area = (bbox2['x_bottom_right'] - bbox2['x_top_left']) * (bbox2['y_bottom_right'] - bbox2['y_top_left'])
    
    iou = interArea / float(box1Area + box2Area - interArea)
    return iou


def match_bboxes(reprojected_bboxes, cross_view_data, frame, camera):
    """
    Match subjects between reprojected bounding boxes and cross_view_data based on IoU.
    """
    # Extract bounding boxes for the specific frame and camera from cross_view_data
    cross_view_bboxes = []
    for cluster_name, image_path_list in cross_view_data[frame].items():
        for image_path in image_path_list:
            bbox_info = extract_bbox_info(image_path)
            if int(bbox_info['camera_id']) == camera:
                cross_view_bboxes.append(bbox_info)
                #print(cross_view_bboxes[-1])
    
    matches = []
    for i, rep_bbox in enumerate(reprojected_bboxes):
        max_iou = 0
        max_j = -1
        for j, cross_view_bbox in enumerate(cross_view_bboxes):
            iou = compute_iou(rep_bbox, cross_view_bbox['bbox'])
            if iou > max_iou:
                max_iou = iou
                max_j = j
        
        if max_iou > 0.4:  # Threshold can be adjusted
            matches.append((i, max_j))
    
    ## matches return the indices of bboxes, and cross_view_bboxes
    return matches, cross_view_bboxes


def match_two_bbox_sets(bbox_projected_world, bbox_detected_camera, cross_view_data, frame, camera1=None, camera2=None):
    """
    Match two sets of bounding boxes through the cross_view_data.
    """
    # Step 4a: Match Set A with cross_view_matching['cross_view_match']
    matches_A, _ = match_bboxes(bbox_projected_world, cross_view_data, frame, camera1)
    #print(matches_A)
    
    # Step 4b: Match Set B with cross_view_matching['cross_view_match']
    matches_B, cross_view_bboxes = match_bboxes(bbox_detected_camera, cross_view_data, frame, camera2)
    #print(matches_B)

    # Step 4c: Use matched pairs A and B to establish the correspondence between Set A and Set B
    final_matches = []
    for idxA, idx_cross_view_A in matches_A:
        for idxB, idx_cross_view_B in matches_B:
            if idx_cross_view_A == idx_cross_view_B:
                final_matches.append((idxA, idxB))


    # print("Final Matches: ", final_matches)
    
    ## Return the matches
    ## (match indices of setA_bboxes, match indices of setB_bboxes)
    return final_matches


