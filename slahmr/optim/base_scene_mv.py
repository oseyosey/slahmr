import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from body_model import SMPL_JOINTS, KEYPT_VERTS, smpl_to_openpose, run_smpl
from geometry.rotation import (
    rotation_matrix_to_angle_axis,
    angle_axis_to_rotation_matrix,
)
from util.logger import Logger
from util.tensor import move_to, detach_all

from .helpers import estimate_initial_trans
from .params import CameraParams
from .multi_view_tools import *

import cv2

from scipy.optimize import linear_sum_assignment
from collections import Counter



J_BODY = len(SMPL_JOINTS) - 1  # no root


class BaseSceneModelMV(nn.Module):
    """
    Scene model of sequences of human poses.
    All poses are in their own INDEPENDENT camera reference frames.
    A basic class mostly for testing purposes.

    Parameters:
        batch_size:  number of sequences to optimize
        seq_len:     length of the sequences
        body_model_multi:  SMPL body model for each view
        pose_prior:  VPoser model
        fit_gender:  gender of model (optional)
    """

    def __init__(
        self,
        batch_size,
        seq_len,
        body_model_multi,
        pose_prior,
        fit_gender="neutral",
        use_init=False,
        opt_cams=False,
        opt_scale=False, ## assume the scale is fixed, static camera. 
        rt_pairs=None, # list of (R, t) pairs for each view in numpy
        rt_pairs_tensor=None, # transformed version of rt_pairs in tensor format
        view_nums=0, # number of views
        **kwargs,
    ):
        super().__init__()
        B, T = batch_size, seq_len
        self.batch_size = batch_size
        self.seq_len = seq_len

        self.body_model_multi = body_model_multi
        self.body_model = body_model_multi[0] # Requires update_batch to be called before initialize
        self.fit_gender = fit_gender

        self.pose_prior = pose_prior
        self.latent_pose_dim = self.pose_prior.latentD

        self.num_betas = self.body_model.bm.num_betas

        # Returns the indices of the permutation that maps SMPL to OpenPose # 
        self.smpl2op_map = smpl_to_openpose(
            self.body_model.model_type,
            use_hands=False,
            use_face=False,
            use_face_contour=False,
            openpose_format="coco25",
        )

        self.use_init = use_init
        print("USE INIT", use_init)
        self.opt_scale = opt_scale
        self.opt_cams = opt_cams
        print("OPT SCALE", self.opt_scale)
        print("OPT CAMERAS", self.opt_cams)

        self.params = CameraParams(batch_size) # Requires update_batch to be called before initialize

        self.rt_pairs = rt_pairs
        self.view_nums = view_nums

        self.rt_pairs_tensor = None
        self.pairing_info = None

    def initialize(self, obs_data_list, cam_data):
        """
        Intializating Multi-view people in the world
        obs_data_list: list of observed data in data loader format
        """
        Logger.log("Initializing scene model with observed data from multiple views")

        breakpoint()

        # initialize cameras
        self.params.set_cameras(
            cam_data,
            opt_scale=self.opt_scale,
            opt_cams=self.opt_cams,
            opt_focal=self.opt_cams,
        )

        ### Multi-view Set-Up ### 
        init_pose_list, init_pose_latent_list, init_betas_list, init_trans_list, init_rot_list, pred_smpl_data_list = [], [], [], [], [], []
        rt_pairs_tensor = []
        for num_view in range(len(obs_data_list)):
            # initialize body params
            obs_data = obs_data_list[num_view]
            B, T = obs_data['joints2d'].shape[0], self.seq_len
            device = next(iter(cam_data.values())).device
            init_betas = torch.zeros(B, self.num_betas, device=device)

            if self.use_init and "init_body_pose" in obs_data:
                init_pose = obs_data["init_body_pose"][:, :, :J_BODY, :]
                init_pose_latent = self.pose2latent(init_pose)
            else:
                init_pose = torch.zeros(B, T, J_BODY, 3, device=device)
                init_pose_latent = torch.zeros(B, T, self.latent_pose_dim, device=device)

            # transform into world frame (T, 3, 3), (T, 3)
            if num_view == 0: # world frame
                R_w2c, t_w2c = cam_data["cam_R"], cam_data["cam_t"]
                R_c2w = R_w2c.transpose(-1, -2)
                t_c2w = -torch.einsum("tij,tj->ti", R_c2w, t_w2c)
            else: # camera frame, require multi-view rt pairs,
                R_w2c_vec, t_w2c = self.rt_pairs[num_view]
                t_w2c = torch.from_numpy(t_w2c).to(device)
                R_w2c = cv2.Rodrigues(R_w2c_vec)[0]
                R_w2c =  torch.from_numpy(R_w2c).to(device)

                t_w2c = t_w2c.squeeze().repeat(T, 1).float()
                R_w2c = R_w2c.unsqueeze(0).repeat(T, 1, 1).float()

                R_c2w = R_w2c.transpose(-1, -2)
                t_c2w = -torch.einsum("tij,tj->ti", R_c2w, t_w2c)
 
            if self.use_init and "init_root_orient" in obs_data:
                init_rot = obs_data["init_root_orient"]  # (B, T, 3)
                init_rot_mat = angle_axis_to_rotation_matrix(init_rot)
                init_rot_mat = torch.einsum("tij,btjk->btik", R_c2w, init_rot_mat) # Transform init_root_orient from camera frame to world frame
                init_rot = rotation_matrix_to_angle_axis(init_rot_mat)
            else:
                init_rot = (
                    torch.tensor([np.pi, 0, 0], dtype=torch.float32)
                    .reshape(1, 1, 3)
                    .repeat(B, T, 1)
                )

            init_trans = torch.zeros(B, T, 3, device=device)
            if self.use_init and "init_trans" in obs_data:
                # must offset by the root location before applying camera to world transform
                pred_data = self.pred_smpl(init_trans, init_rot, init_pose, init_betas)
                root_loc = pred_data["joints3d"][..., 0, :]  # (B, T, 3)
                init_trans = obs_data["init_trans"]  # (B, T, 3)
                ## here it's calculating the root translation in the world coordiante frame using R(R_c2w) and T(t_c2w) with init_trans (Root translation in the camera frame)
                init_trans = (
                    torch.einsum("tij,btj->bti", R_c2w, init_trans + root_loc)
                    + t_c2w[None]
                    - root_loc
                )
            else:
                # initialize trans with reprojected joints
                pred_data = self.pred_smpl(init_trans, init_rot, init_pose, init_betas)
                init_trans = estimate_initial_trans( ## use focal length and bone lengths to approximate distance from camera or Root translation in the world coordiante frame. 
                    init_pose,
                    pred_data["joints3d_op"], ## Here the joints3d_op seems to be joints 3d in the world coordiante frame? ##TODO: Joints3d_op + init_trans seems to be the actual world frame 3D info.
                    obs_data["joints2d"],
                    obs_data["intrins"][:, 0],
                )

            pred_smpl_data_list.append(pred_data)
            rt_pairs_tensor.append((R_w2c, t_w2c))

            init_pose_latent_list.append(init_pose_latent)
            init_pose_list.append(init_pose)
            init_betas_list.append(init_betas)
            init_trans_list.append(init_trans)
            init_rot_list.append(init_rot)


        self.rt_pairs_tensor = rt_pairs_tensor
        


        ## 先把所有需要的东西存下来，放在Jupyter Notebook里操作
        import pickle
        # Create a dictionary with the variables as keys
        data_to_store = {
            "init_pose_latent_list": init_pose_latent_list,  
            "init_pose_list": init_pose_list,          
            "init_betas_list": init_betas_list,        
            "init_trans_list": init_trans_list,        
            "init_rot_list": init_rot_list,         
            "pred_smpl_data_list": pred_smpl_data_list,   
            "obs_data_list": obs_data_list        
        }
        # Path to store the pickle file
        pickle_file_path = 'stich_world_data.pickle' # save to here '/share/kuleshov/jy928/slahmr/outputs/logs/images-val/2023-11-06/Camera0-all-shot-0-1-50'
        with open(pickle_file_path, 'wb') as handle:
            pickle.dump(data_to_store, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print("Data saved to pickle file")


        breakpoint()
        # Obtain world smpl parameters from multi-view smpl parameters through stitching. 
        init_pose_latent_world, init_betas_world, init_trans_world, init_rot_world, matching_obs_data = self.stitch_world_tracklet(init_pose_latent_list, init_betas_list, 
                                                                                                                                   init_trans_list, init_rot_list, pred_smpl_data_list, obs_data_list, device)
        self.params.set_param("latent_pose", init_pose_latent_world) ## pose in the world frame
        self.params.set_param("betas", init_betas_world) ## beta in the world frame
        self.params.set_param("trans", init_trans_world) ## root translation in the world frame
        self.params.set_param("root_orient", init_rot_world) ## root orientation in the world frame 



        # Create a dictionary with the variables as keys
        data_to_store_stitched = {
            "init_pose_latent_world": init_pose_latent_world,  
            "init_pose_world":  self.latent2pose(init_pose_latent_world), 
            "init_betas_world": init_betas_world,         
            "init_trans_world": init_trans_world,        
            "init_rot_world": init_rot_world   
        }
        pickle_file_path = 'stich_world_data_stitched.pickle' # save to here '/share/kuleshov/jy928/slahmr/outputs/logs/images-val/2023-11-06/Camera0-all-shot-0-1-50'
        with open(pickle_file_path, 'wb') as handle:
            pickle.dump(data_to_store_stitched, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print("Stitched Data saved to pickle file")


        return rt_pairs_tensor


    def stitch_world_tracklet(self, init_pose_latent_list, init_betas_list, init_trans_list, init_rot_list, pred_smpl_data_list, obs_data_list, device):
        """
        Stitch tracklets from each view (already transformed from camera to world) into one global tracklet in the world frame

        Input:
        init_pose_latent_list: list of latent pose in the camera frame
        init_betas_list: list of betas in the camera frame
        init_trans_list: list of root translation in the camera frame
        init_rot_list: list of root orientation in the camera frame
        pred_smpl_data_list: list of smpl data in the camera frame
        obs_data_list: list of observed data in data loader format

        Output:
        init_pose_latent_world: latent pose in the world frame
        init_betas_world: betas in the world frame
        init_trans_world: root translation in the world frame
        init_rot_world: root orientation in the world frame
        matching_obs_data: matching observed data between data loader for each view and world frame tracklet data
        """  

        # Initialize 2D list with the first view
        view_init = 0
        pose_latent_2D_list, betas_2D_list, trans_2D_list, rot_2D_list  = self.initialize_2D_smpl_param_lists(init_pose_latent_list, init_betas_list, init_trans_list, init_rot_list, view_init=view_init)
        

        # Initialize a dictionary to store the pairing information for each view
        pairing_info_all_views = {}

        for t_ in range(self.seq_len):
            for view in range(1, self.view_nums): 
                selected_tracks, selected_tracks_indices, has_first_appearance = self.select_first_appearance(init_rot_list, view, t_)
                if has_first_appearance:
                    # match selected tracklet with world tracklet (SLAHMR Results)

                    set_counter = Counter() # for voting
                    for look_foward in range(0, 25): # look forward 24 frames
                        if t_+look_foward < self.seq_len:
                            ## Select SMPL prediction data in world frame (6890, 3)
                            selected_tracks_smpl_3d_points = pred_smpl_data_list[view]['points3d'][selected_tracks_indices][:,t_+look_foward, :, :]
                            matched_subjects, unmatched_subjects, matched_indices_pair, unmatched_indices_world, unmatched_indices_camera  = self.match_subjects_3d_points(pred_smpl_data_list[0]['points3d'][:,t_+look_foward, :, :].cpu().detach().numpy(), 
                                                                                                                                                                       selected_tracks_smpl_3d_points.cpu().detach().numpy()) 

                            # Map the matched and unmatched indices back to pred_smpl_data_list[view]
                            # Replace camera_index with camera_index_global in matched_indices_pair
                            matched_indices_pair_orig = [(world_index, selected_tracks_indices[camera_index]) for world_index, camera_index in matched_indices_pair]

                            unmatched_indices_camera_orig = [selected_tracks_indices[camera_index] for camera_index in unmatched_indices_camera ]

                            # Combine matched_indices_pair_orig and unmatched_indices_camera_orig
                            combined_set = set(matched_indices_pair_orig + [(index,) for index in unmatched_indices_camera_orig])

                            # Convert the set to a frozenset and count it
                            set_counter[frozenset(combined_set)] += 1

                    # Get the most common set
                    most_common_set = set_counter.most_common(1)[0][0]

                    # Convert the most common frozenset back to a set
                    most_common_set = set(most_common_set)

                    # Convert the set back to matching_pairs, and unmatched_indices
                    matched_indices_pair_orig = [pair for pair in most_common_set if len(pair) == 2]
                    unmatched_indices_camera_orig = [pair[0] for pair in most_common_set if len(pair) == 1]
                    
                    # Create a dictionary to store the pairing information for this view
                    pairing_info_per_view = {}

                    # For the matching cases #
                    # append selected_tracks_indices to the 2D list (B[i, i+1], T, D)
                    for match_pairs in matched_indices_pair_orig:
                        world_index, camera_index = match_pairs
                        # Get the corresponding latent pose, betas, root translation, and root orientation 
                        pose_latent_2D_list[world_index].append(init_pose_latent_list[view][camera_index])
                        betas_2D_list[world_index].append(init_betas_list[view][camera_index])
                        trans_2D_list[world_index].append(init_trans_list[view][camera_index])
                        rot_2D_list[world_index].append(init_rot_list[view][camera_index])

                        first_appearance = t_
                        last_apperance = self.find_last_apperance(init_rot_list[view][camera_index])

                        pairing_info_per_view[world_index] = (camera_index, first_appearance, last_apperance)

                    # For the unmatched cases #
                    # Append a new list to the 2D list (B+1, T, D)
                    for camera_index in unmatched_indices_camera_orig:
                        pose_latent_2D_list.append([init_pose_latent_list[view][camera_index]])
                        betas_2D_list.append([init_betas_list[view][camera_index]])
                        trans_2D_list.append([init_trans_list[view][camera_index]])
                        rot_2D_list.append([init_rot_list[view][camera_index]])

                        world_index = len(rot_2D_list) - 1 # index starts from 0

                        first_appearance = t_
                        last_apperance = self.find_last_apperance(init_rot_list[view][camera_index])

                        pairing_info_per_view[world_index] = (camera_index, first_appearance, last_apperance)


                    pairing_info_all_views[view] = pairing_info_per_view


            # transform (combining / averaging of information) the 2D list to updated SMPL paramters in the world frame (using weighted average, possibly confidence score)
            # (Update the world tracklet pose parameteres with the corresponding latent pose, betas, root translation, and root orientation)
            init_pose_latent_world, init_betas_world, init_trans_world, init_rot_world = self.merge_2D_smpl_param_lists(pose_latent_2D_list, betas_2D_list, trans_2D_list, rot_2D_list)

            return init_pose_latent_world, init_betas_world, init_trans_world, init_rot_world, pairing_info_all_views



    def merge_2D_smpl_param_lists(self, pose_latent_2D_list, betas_2D_list, trans_2D_list, rot_2D_list):
        """
        Merging 2D SMPL parameters from multiple views into one global tracklet in the world frame

        Input:
        """
        ## Merge trans_2D_list  ##
        init_trans_world = calculate_weighted_averages_trans_2D(trans_2D_list)

        ## Merge betas_2D_list ##
        init_betas_world = calculate_weighted_averages_betas_2D(betas_2D_list)

        ## Merge rot_2D_list ##
        init_rot_world = calculate_weighted_averages_rot_2D(rot_2D_list)

        ## Merge pose_2D_list ##
        init_pose_latent_world = self.calculate_weighted_averages_pose_latent_2D(pose_latent_2D_list)

        return init_pose_latent_world, init_betas_world, init_trans_world, init_rot_world




    def match_subjects_3d_points(self, pred_smpl_data_list, selected_tracks_smpl_3d_points):
        """
        Hungarian Matching of 3D points (6890, 3) between two tracklets 

        Input:
        pred_smpl_data_list: list of SMPL data in the world frame
        selected_tracks_smpl_3d_points: list of SMPL data in the world frame (camera transformed)

        Output:
        matched_subjects: list of matching pairs (world, camera)
        unmatched_subjects: list of unmatched subjects in the world frame
        matched_indices_pairs: lists of matching pairs (world_index, camera_index)
        unmatched_indices_world: list of unmatched indices in the world frame
        unmatched_indices_camera: list of unmatched indices in the camera frame
        """
        # Convert the lists to numpy arrays for easier manipulation
        pred_smpl_data_array = np.array(pred_smpl_data_list)
        selected_tracks_array = np.array(selected_tracks_smpl_3d_points)

        # Compute the cost matrix
        cost_matrix = np.zeros((len(pred_smpl_data_array), len(selected_tracks_array)))
        for i in range(len(pred_smpl_data_array)):
            for j in range(len(selected_tracks_array)):
                # TODO: Compute the cost between pred_smpl_data_array[i] and selected_tracks_array[j]
                cost = self.compute_cost_l2(pred_smpl_data_array[i], selected_tracks_array[j])
                cost_matrix[i, j] = cost

        # Use the Hungarian algorithm to find the optimal assignment
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        # Get the matched and unmatched subjects
        matched_subjects = [(pred_smpl_data_array[i], selected_tracks_array[j]) for i, j in zip(row_indices, col_indices)]
        unmatched_subjects = [pred_smpl_data_array[i] for i in range(len(pred_smpl_data_array)) if i not in row_indices]

        # Get the indices of the matched and unmatched subjects
        matched_indices_pairs = list(zip(row_indices, col_indices))
        unmatched_indices_world = [i for i in range(len(pred_smpl_data_array)) if i not in row_indices]
        unmatched_indices_camera = [j for j in range(len(selected_tracks_array)) if j not in col_indices]

        return matched_subjects, unmatched_subjects, matched_indices_pairs, unmatched_indices_world, unmatched_indices_camera
    

    def compute_cost_l2(self, pred_smpl_data, selected_track):
        """
        Input:
        pred_smpl_data: SMPL data in the world frame
        selected_track: SMPL data in the world frame (camera transformed)

        Output:
        Compute the L2 distance between each pair of 3D points
        """
        cost = np.sum(np.linalg.norm(pred_smpl_data - selected_track, axis=1))

        return cost


    def initialize_2D_smpl_param_lists(self, init_pose_latent_list, init_betas_list, init_trans_list, init_rot_list, view_init=0):
        # Initialize the 2D lists with the first view's data
        pose_latent_2D_list = [[init_pose_latent] for init_pose_latent in init_pose_latent_list[view_init]]
        betas_2D_list = [[init_betas] for init_betas in init_betas_list[view_init]]
        trans_2D_list = [[init_trans] for init_trans in init_trans_list[view_init]]
        rot_2D_list = [[init_rot] for init_rot in init_rot_list[view_init]]

        return pose_latent_2D_list, betas_2D_list, trans_2D_list, rot_2D_list


    def select_first_appearance(self, init_rot_list, view_num, t_):
        """
        Input:
        init_rot_list: list of root orientation in the camera frame
        view_num: the view number
        t_: the current frame

        Output:
        selected_tracks: list of selected tracks (can turn into numpy)
        """
        # Access the specific view
        view = init_rot_list[view_num]

        # Initialize a list to store the selected tracks
        selected_tracks = []
        selected_tracks_indices = []

        # Initialize a boolean to indicate whether there is a first appearance
        has_first_appearance = False

        # Iterate over the tracks
        index = 0
        for track in view:
            # Check if the current frame is the first non-zero frame
            if (np.all(track[:t_].to('cpu').numpy() == 0) and np.all(track[t_].to('cpu').numpy() != 0)):
                view_num += 1
                selected_tracks.append(track)
                selected_tracks_indices.append(index)
                index += 1
                has_first_appearance = True

        # Convert the list of selected tracks to a numpy array
        # selected_tracks = np.array(selected_tracks)

        return selected_tracks, selected_tracks_indices, has_first_appearance


    def find_last_apperance(self, init_rot):
        """
        Input:
        init_rot: root orientation in the camera frame

        Output:
        t_: last apperance frame 
        """
        for t_ in range(self.seq_len-1):
            if np.all(init_rot[t_].to('cpu').numpy() != 0) and np.all(init_rot[t_+1].to('cpu').numpy() == 0):
                return t_
            elif t_+1 == self.seq_len-1:
                return t_+1
                
    
    def calculate_weighted_averages_pose_latent_2D(self, pose_latent_2D_list):
        """
        Averages the pose latents in a 2D list with weighted averaging on the corresponding pose representation.
        Each latent tensor is first converted to pose, then weighted averaging is performed, and the result is converted back to latent representation.
        
        Parameters:
        pose_latent_2D_list (list of list of torch.Tensor): A list where each element is a sublist containing tensors of pose latent representation.

        Returns:
        torch.Tensor: A tensor containing the weighted averages of the pose latents for each sublist.
        """
        weights_list = []

        # Create weights, giving the first element a weight of 2 and the rest a weight of 1
        for latent_list in pose_latent_2D_list:
            weights = [2] + [1] * (len(latent_list) - 1)
            weights_list.append(weights)

        weighted_averages_latent = []

        # Process each sublist
        for latent_list, weights in zip(pose_latent_2D_list, weights_list):
            poses = []
            # Convert each latent tensor to a pose tensor
            for latent in latent_list:
                # Add a batch dimension B
                latent_unsqz = latent.unsqueeze(0)
                pose = self.latent2pose(latent_unsqz)
                # Remove the batch dimension after conversion
                pose_sqz = pose.squeeze(0)
                poses.append(pose_sqz)

            # Perform weighted average on poses
            weighted_sum = sum(p * w for p, w in zip(poses, weights))
            weighted_average_pose = weighted_sum / sum(weights)

            # Add the batch dimension before converting back to latent
            weighted_average_pose_unsqz = weighted_average_pose.unsqueeze(0)
            weighted_average_latent = self.pose2latent(weighted_average_pose_unsqz)
            # Remove the batch dimension to match the original size
            weighted_average_latent_sqz = weighted_average_latent.squeeze(0)

            weighted_averages_latent.append(weighted_average_latent_sqz)

        # Combine all weighted average latents into a single tensor
        weighted_averages_latent_tensor = torch.stack(weighted_averages_latent, dim=0)

        return weighted_averages_latent_tensor




    def update_pairing_info(self, pairing_info_all_views):
        self.pairing_info = pairing_info_all_views

    def update_batch(self, batch_size_new, body_model_new):
        """
        Update batch_size:  number of sequences to optimize.
        Update body_model: SMPL body model for global view.
        Multi
        """
        self.batch_size = batch_size_new
        self.params = CameraParams(batch_size_new)
        self.body_model = body_model_new


    def get_optim_result(self, **kwargs):
        """
        Collect predicted outputs (latent_pose, trans, root_orient, betas, body pose) into dict
        """
        res = self.params.get_dict()
        if "latent_pose" in res:
            res["pose_body"] = self.latent2pose(self.params.latent_pose).detach()

        # add the cameras
        res["cam_R"], res["cam_t"], _, _ = self.params.get_cameras()
        res["intrins"] = self.params.intrins
        return {"world": res}

    def latent2pose(self, latent_pose):
        """
        Converts VPoser latent embedding to aa body pose.
        latent_pose : B x T x D
        body_pose : B x T x J*3
        """
        B, T, _ = latent_pose.size()
        d_latent = self.pose_prior.latentD
        latent_pose = latent_pose.reshape((-1, d_latent))
        body_pose = self.pose_prior.decode(latent_pose, output_type="matrot")
        body_pose = rotation_matrix_to_angle_axis(
            body_pose.reshape((B * T * J_BODY, 3, 3))
        ).reshape((B, T, J_BODY * 3))
        return body_pose

    def pose2latent(self, body_pose):
        """
        Encodes aa body pose to VPoser latent space.
        body_pose : B x T x J*3
        latent_pose : B x T x D
        """
        B, T = body_pose.shape[:2]
        body_pose = body_pose.reshape((-1, J_BODY * 3))
        latent_pose_distrib = self.pose_prior.encode(body_pose)
        d_latent = self.pose_prior.latentD
        latent_pose = latent_pose_distrib.mean.reshape((B, T, d_latent))
        return latent_pose

    def pred_smpl(self, trans, root_orient, body_pose, betas, num_view=None):
        """
        Forward pass of the SMPL model and populates pred_data accordingly with
        joints3d, verts3d, points3d.

        trans : B x T x 3
        root_orient : B x T x 3
        body_pose : B x T x J*3
        betas : B x D
        """
        if num_view is None:
            body_model = self.body_model
        else:
            body_model = self.body_model_multi[num_view]

        smpl_out = run_smpl(body_model, trans, root_orient, body_pose, betas)
        joints3d, points3d = smpl_out["joints"], smpl_out["vertices"]

        # select desired joints and vertices
        joints3d_body = joints3d[:, :, : len(SMPL_JOINTS), :]
        joints3d_op = joints3d[:, :, self.smpl2op_map, :]
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

    def pred_params_smpl(self, reproj=True):
        body_pose = self.latent2pose(self.params.latent_pose)
        pred_data = self.pred_smpl(
            self.params.trans, self.params.root_orient, body_pose, self.params.betas
        )

        return pred_data
