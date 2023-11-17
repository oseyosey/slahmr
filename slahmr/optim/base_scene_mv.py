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

from util.loaders import (
    load_smpl_body_model
)

from .helpers import estimate_initial_trans
from .params import CameraParamsMV
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
        pairing_info =None,
        path_body_model = None,
        body_model_stitch = None,
        opt_scale_mv=False,
        opt_cams_mv=True,
        opt_focal_mv=True,
        **kwargs,
    ):
        super().__init__()
        B, T = batch_size, seq_len
        self.batch_size = batch_size
        self.seq_len = seq_len

        self.body_model_multi = body_model_multi
        
        ## * Address Motion Prior Issue ##
        if body_model_stitch is None:
            self.body_model = body_model_multi[0] # Requires update_batch to be called before initialize
        else:
            self.body_model = body_model_stitch

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

        self.params = CameraParamsMV(batch_size) #*  Requires update_batch to be called before initialize

        self.rt_pairs = rt_pairs
        self.view_nums = view_nums

        self.rt_pairs_tensor = rt_pairs_tensor
        self.pairing_info = pairing_info
        self.path_body_model = path_body_model

        #* Camera and Focal length update #
        self.opt_scale_mv = opt_scale_mv
        self.opt_cams_mv = opt_cams_mv
        self.opt_focal_mv = opt_focal_mv
        print("OPT MULTI-VIEW Scale", self.opt_scale_mv)
        print("OPT MULTI-VIEW FOCAL", self.opt_cams_mv)
        print("OPT MULTI-VIEW CAMERAS", self.opt_focal_mv)


    def initialize(self, obs_data_list, cam_data, slahmr_data_init, rt_pairs_tensor, debug=False):
        """
        Intializating Multi-view people in the world
        obs_data_list: list of observed data in data loader format
        slahmr_data_init: SLAHMR results in the world frame in dictionary format
        """
        Logger.log("Initializing scene model with observed data from multiple views")

        # initialize cameras
        self.params.set_cameras(
            cam_data,
            opt_scale=self.opt_scale,
            opt_cams=self.opt_cams,
            opt_focal=self.opt_cams,
        )

        #* initialize multi-view cameras
        self.params.set_cameras_mv(
            cam_data,
            rt_pairs_tensor,
            self.view_nums,
            opt_scale_mv=self.opt_scale_mv,
            opt_cams_mv=self.opt_cams_mv,
            opt_focal_mv=self.opt_focal_mv,
        )
        

        ### Multi-view Set-Up ### 
        init_pose_list, init_pose_latent_list, init_betas_list, init_trans_list, init_rot_list, pred_smpl_data_list, init_appe_list = [], [], [], [], [], [], []
        rt_pairs_tensor = [] ##* List of (R, t) pairs for each view equivalent to cam_data["cam_R"], cam_data["cam_t"] format.
        for num_view in range(len(obs_data_list)):
            # initialize body params
            obs_data = obs_data_list[num_view]
            B, T = obs_data['joints2d'].shape[0], self.seq_len
            device = next(iter(cam_data.values())).device
            
            init_betas = torch.zeros(B, self.num_betas, device=device)

            if self.use_init and num_view == 0: # Appending SLAHMR Results
                init_pose = slahmr_data_init["pose_body"].view(self.batch_size, self.seq_len, 21, 3) # origionally (B, T, 63)
                init_pose = init_pose[:, :, :J_BODY, :]
                init_pose_latent = self.pose2latent(init_pose)
            elif self.use_init and "init_body_pose" in obs_data:
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
 
            if self.use_init and num_view == 0: ## Appending SLAHMR Results
                init_rot = slahmr_data_init["root_orient"]
                init_rot_mat = angle_axis_to_rotation_matrix(init_rot)
                init_rot_mat = torch.einsum("tij,btjk->btik", R_c2w, init_rot_mat)
                init_rot = rotation_matrix_to_angle_axis(init_rot_mat)
            elif self.use_init and "init_root_orient" in obs_data:
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
            if self.use_init and num_view == 0: ## Appending SLAHMR Results
                pred_data = self.pred_smpl(init_trans, init_rot, init_pose, init_betas, num_view) # Note that here init_trans is zeros tensors
                root_loc = pred_data["joints3d"][..., 0, :]  # (B, T, 3)
                init_trans = obs_data["init_trans"]  # (B, T, 3)
                init_trans = (
                    torch.einsum("tij,btj->bti", R_c2w, init_trans + root_loc)
                    + t_c2w[None]
                    - root_loc
                )
            elif self.use_init and "init_trans" in obs_data:
                # must offset by the root location before applying camera to world transform
                pred_data = self.pred_smpl(init_trans, init_rot, init_pose, init_betas, num_view)
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
                    pred_data["joints3d_op"], ## Here the joints3d_op seems to be joints 3d in the world coordiante frame? ##TODO: Joints3d_op + init_trans seems to be the actual world frame 3D inf?
                    obs_data["joints2d"],
                    obs_data["intrins"][:, 0],
                )

            ## Here we are using the 4D Human results for the appearance embedding for view 0 (reference view)
            init_appe = obs_data['init_appe'] if self.use_init and 'init_appe' in obs_data else None


            ## Recompute pred_smpl_data_list in the world frame (with updated init_trans)
            pred_data_world = self.pred_smpl(init_trans, init_rot, init_pose, init_betas, num_view)
            pred_smpl_data_list.append(pred_data_world)
            rt_pairs_tensor.append((R_w2c, t_w2c))

            init_pose_latent_list.append(init_pose_latent)
            init_pose_list.append(init_pose)
            init_betas_list.append(init_betas)
            init_trans_list.append(init_trans)
            init_rot_list.append(init_rot)

            init_appe_list.append(init_appe)

        # Set-up rt_pairs_tensor in our BaseSceneModelMV (no longer needed)
        # self.rt_pairs_tensor = rt_pairs_tensor
        
        if debug:
            ## 先把所有需要的东西存下来，放在Jupyter Notebook里操作
            import pickle

            # Create a dictionary with the variables as keys
            data_to_store = {
                #"init_pose_latent_list": init_pose_latent_list,  
                #"init_pose_list": init_pose_list,          
                #"init_betas_list": init_betas_list,        
                #"init_trans_list": init_trans_list,        
                #"init_rot_list": init_rot_list,         
                "pred_smpl_data_list": pred_smpl_data_list,   
                #"obs_data_list": obs_data_list, 
                # "init_appe_list": init_appe_list       
            }
            # Path to store the pickle file
            pickle_file_path = 'stich_world_data_smpl.pickle' # save to here '/share/kuleshov/jy928/slahmr/outputs/logs/images-val/2023-11-06/Camera0-all-shot-0-1-50'
            with open(pickle_file_path, 'wb') as handle:
                pickle.dump(data_to_store, handle, protocol=pickle.HIGHEST_PROTOCOL)
                print("Stitch Data Init saved to pickle file")

        
        ### Obtain world smpl parameters from multi-view smpl parameters through stitching ###
        init_pose_latent_world, init_betas_world, init_trans_world, init_rot_world, matching_obs_data = self.stitch_world_tracklet(init_pose_latent_list, init_betas_list, 
                                                                                                                                   init_trans_list, init_rot_list, pred_smpl_data_list, init_appe_list, obs_data_list, device)

        #TODO (Solved): init_trans_world, init_rot_world, init_rot_world last subject [4] is all 0s, check why.  Edge cases when num element is = 1
        if debug:
            breakpoint()


        self.params.set_param("latent_pose", init_pose_latent_world) ## pose in the world frame
        self.params.set_param("betas", init_betas_world) ## beta in the world frame
        self.params.set_param("trans", init_trans_world) ## root translation in the world frame
        self.params.set_param("root_orient", init_rot_world) ## root orientation in the world frame 


        ## Obtain new batch size (number of sequences) to optimie 
        b_stitched = init_pose_latent_world.shape[0] # number of sequences to optimize

        ## Update batch size and body model in BaseSceneModelMV
        body_model_stitch, fit_gender = load_smpl_body_model(self.path_body_model, b_stitched * self.seq_len, device=device)
        self.update_batch(b_stitched, body_model_stitch)


        ### Tracking Results Storing ### 
        track_flag = True
        if debug or track_flag:
            import pickle
            pred_data_world = self.pred_smpl(init_trans_world, init_rot_world, self.latent2pose(init_pose_latent_world), init_betas_world)
            # Create a dictionary with the variables as keys
            data_to_store_stitched = {
                "init_pose_latent_world": init_pose_latent_world,  
                "init_pose_world":  self.latent2pose(init_pose_latent_world), 
                "init_betas_world": init_betas_world,         
                "init_trans_world": init_trans_world,        
                "init_rot_world": init_rot_world,
                "init_pred_smpl_data_world": pred_data_world,
                "rt_pairs": self.rt_pairs,
                "rt_pairs_tensor": self.rt_pairs_tensor,    
                "init_cam_data": cam_data, # camera pose (RT) in the world frame for view 0,
                #"init_intrinsics": obs_data_list[0][0]["intrins"], # camera intrinsics for view 0
            }

            pickle_file_path = 'stich_world_data_stitched.pickle' # save to here '/share/kuleshov/jy928/slahmr/outputs/logs/images-val/2023-11-06/Camera0-all-shot-0-1-50'
            with open(pickle_file_path, 'wb') as handle:
                pickle.dump(data_to_store_stitched, handle, protocol=pickle.HIGHEST_PROTOCOL)
                print("Stitched Data saved to pickle file")

        return self.rt_pairs_tensor, matching_obs_data, b_stitched, body_model_stitch #* rt_pairs_tensor no longer needed



    def stitch_world_tracklet(self, init_pose_latent_list, init_betas_list, init_trans_list, init_rot_list, pred_smpl_data_list, init_appe_list, obs_data_list, device, debug=False):
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
        pose_latent_2D_list, betas_2D_list, trans_2D_list, rot_2D_list  = self.initialize_2D_smpl_param_lists(init_pose_latent_list, init_betas_list, init_trans_list, init_rot_list)
        

        # Initialize a dictionary to store the pairing information for each view
        pairing_info_all_views = {}

        for t_ in range(0, self.seq_len-1):
            for view in range(0, self.view_nums): 
                selected_tracks, selected_tracks_indices, has_first_appearance = self.select_first_appearance(obs_data_list, view, t_) ##TODO! 
                if has_first_appearance:
                    # If First Appearance; initialize the track #
                    if (not pose_latent_2D_list) and (not betas_2D_list) and (not trans_2D_list) and (not rot_2D_list):
                        pose_latent_2D_list = [[init_pose_latent] for init_pose_latent in init_pose_latent_list[view][selected_tracks_indices]]
                        betas_2D_list = [[init_betas] for init_betas in init_betas_list[view][selected_tracks_indices]]
                        trans_2D_list = [[init_trans] for init_trans in init_trans_list[view][selected_tracks_indices]]
                        rot_2D_list = [[init_rot] for init_rot in init_rot_list[view][selected_tracks_indices]]

                        pairing_info_per_view = {} # * Create a dictionary to store the pairing information for this view #
                        for world_index in range(0, len(selected_tracks_indices)):
                            camera_index = selected_tracks_indices[world_index]
                            first_appearance = t_
                            last_apperance = self.find_last_apperance(obs_data_list[view]['joints2d'][camera_index])
                            pairing_info_per_view[world_index] = [(camera_index, first_appearance, last_apperance)]
                            pairing_info_all_views[view] = pairing_info_per_view

                    else:
                    # Match selected tracklet with world tracklet (SLAHMR Results) #
                        set_counter = Counter() # for voting

                        # * adaptively generate points3d #
                        b_stitch = len(pose_latent_2D_list) # number of sequences to optimize
                        body_model_stitch, fit_gender = load_smpl_body_model(self.path_body_model, b_stitch * self.seq_len, device=device)
                        init_pose_latent_world, init_betas_world, init_trans_world, init_rot_world = self.merge_2D_smpl_param_lists(pose_latent_2D_list, betas_2D_list, trans_2D_list, rot_2D_list)
                        stitch_smpl_data = self.pred_smpl_adapt(init_trans_world, init_rot_world, self.latent2pose(init_pose_latent_world), init_betas_world, body_model_stitch)

                        look_back = -24
                        look_forward = 24
                        for look_foward in range(look_back, look_forward): # look forward 24 frames
                            offset = 0
                            if t_+look_foward < 0:
                                offset = -look_back

                            if t_+look_foward < self.seq_len:
                                ## Select SMPL prediction data in world frame (6890, 3)
                                #selected_tracks_smpl_3d_points = pred_smpl_data_list[view]['points3d'][selected_tracks_indices][:,t_+look_foward, :, :] #! We shouldn't select any tracks here to perform matching, because it will cut the real match
                                matched_subjects, unmatched_subjects, matched_indices_pair, unmatched_indices_world, unmatched_indices_camera  = self.match_subjects_3d_points_appe(stitch_smpl_data['points3d'][:,t_+look_foward+offset, :, :].cpu().detach().numpy(), # TODO: we want to use the most current 3D points 3d instead of SLAHMR results
                                                                                                                                                                        pred_smpl_data_list[view]['points3d'][:,t_+look_foward+offset, :, :].cpu().detach().numpy(), 
                                                                                                                                                                        init_appe_list[0 ][:,t_+look_foward+offset, :].cpu().detach().numpy(),
                                                                                                                                                                        init_appe_list[view][:,t_+look_foward+offset, :].cpu().detach().numpy())
                                if debug:
                                    print(f"Frame {t_}: unmatched_indices_world at view {view}", unmatched_indices_world)
                                    print(f"Frame {t_}: unmatched_indices_camera at view {view}", unmatched_indices_camera)

                                # Map the matched and unmatched indices back to pred_smpl_data_list[view]
                                # Replace camera_index with camera_index_global in matched_indices_pair
                                ## TODO: New way of matching, it will only select matchings if the matched indices are in the selected_tracks_indices (first appearance)
                                matched_indices_pair_orig = []
                                for world_index, camera_index in matched_indices_pair:
                                    if camera_index in selected_tracks_indices:
                                        matched_indices_pair_orig.append((world_index, camera_index))
                                # [(world_index, selected_tracks_indices[camera_index]) for world_index, camera_index in matched_indices_pair]

                                unmatched_indices_camera_orig = []
                                for camera_index in unmatched_indices_camera:
                                    if camera_index in selected_tracks_indices:
                                        unmatched_indices_camera_orig.append(camera_index)
                                
                                # unmatched_indices_camera_orig = [selected_tracks_indices[camera_index] for camera_index in unmatched_indices_camera ]
                                if debug:
                                    print(f"Frame {t_}: matched_indices_pair_orig at view {view}", matched_indices_pair_orig)
                                    print(f"Frame {t_}: unmatched_indices_world_orig at view {view}", unmatched_indices_camera_orig)


                                # Combine matched_indices_pair_orig and unmatched_indices_camera_orig
                                combined_set = set(matched_indices_pair_orig + [(index,) for index in unmatched_indices_camera_orig])

                                # Convert the set to a frozenset and count it
                                set_counter[frozenset(combined_set)] += 1

                        # Get the most common set
                        most_common_set = set_counter.most_common(1)[0][0]

                        # Convert the most common frozenset back to a set
                        most_common_set = set(most_common_set)

                        print(f"Frame {t_}: most_common_set at view {view}", most_common_set)

                        # Convert the set back to matching_pairs, and unmatched_indices
                        matched_indices_pair_orig = [pair for pair in most_common_set if len(pair) == 2]
                        unmatched_indices_camera_orig = [pair[0] for pair in most_common_set if len(pair) == 1]


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
                            last_apperance = self.find_last_apperance(obs_data_list[view]['joints2d'][camera_index])
                            if debug:
                                print(f"Frame {t_}: first_appearance for view {view} at {camera_index}", first_appearance)
                                print(f"Frame {t_}: last_apperance for {view} at {camera_index}", last_apperance)

                            if view in pairing_info_all_views.keys():
                                if world_index in pairing_info_all_views[view].keys():
                                    pairing_info_all_views[view][world_index].append((camera_index, first_appearance, last_apperance))
                                else:
                                    pairing_info_all_views[view][world_index] = [(camera_index, first_appearance, last_apperance)]
                            else:
                                pairing_info_all_views[view] = {}
                                pairing_info_all_views[view][world_index] = [(camera_index, first_appearance, last_apperance)]


                        # For the unmatched cases #
                        # Append a new list to the 2D list (B+1, T, D)
                        for camera_index in unmatched_indices_camera_orig:
                            pose_latent_2D_list.append([init_pose_latent_list[view][camera_index]])
                            betas_2D_list.append([init_betas_list[view][camera_index]])
                            trans_2D_list.append([init_trans_list[view][camera_index]])
                            rot_2D_list.append([init_rot_list[view][camera_index]])

                            world_index = len(rot_2D_list) - 1 # index starts from 0

                            first_appearance = t_
                            last_apperance = self.find_last_apperance(obs_data_list[view]['joints2d'][camera_index])

                            if view in pairing_info_all_views.keys():
                                if world_index in pairing_info_all_views[view].keys():
                                    pairing_info_all_views[view][world_index].append((camera_index, first_appearance, last_apperance))
                                else:
                                    pairing_info_all_views[view][world_index] = [(camera_index, first_appearance, last_apperance)]
                            else:
                                pairing_info_all_views[view] = {}
                                pairing_info_all_views[view][world_index] = [(camera_index, first_appearance, last_apperance)]


        # transform (combining / averaging of information) the 2D list to updated SMPL paramters in the world frame (using weighted average, possibly confidence score)
        # (Update the world tracklet pose parameteres with the corresponding latent pose, betas, root translation, and root orientation)
        init_pose_latent_world, init_betas_world, init_trans_world, init_rot_world = self.merge_2D_smpl_param_lists(pose_latent_2D_list, betas_2D_list, trans_2D_list, rot_2D_list)

        # Update the pairing information into Base Scene Model
        self.update_pairing_info(pairing_info_all_views)

        return init_pose_latent_world, init_betas_world, init_trans_world, init_rot_world, pairing_info_all_views



    def merge_2D_smpl_param_lists(self, pose_latent_2D_list, betas_2D_list, trans_2D_list, rot_2D_list, avg_pose = True):
        """
        Merging 2D SMPL parameters from multiple views into one global tracklet in the world frame

        Input:

        """
        ## Hyperparameters Settings ##
        trans_alpha = 75.0
        betas_alpha = 50.0
        rot_alpha = 99.0
        pose_alpha = 50.0


        ## Merge trans_2D_list  ##
        init_trans_world = calculate_weighted_averages_trans_2D(trans_2D_list, first_element_weight_percentage=trans_alpha)

        ## Merge betas_2D_list ##
        init_betas_world = calculate_weighted_averages_betas_2D(betas_2D_list, first_element_weight_percentage=betas_alpha)

        ## Merge rot_2D_list ##
        if avg_pose:
            ## Average Root Orientation, porential problem! Because ground plane is not established for other view! ##
            init_rot_world = calculate_weighted_averages_rot_2D(rot_2D_list, first_element_weight_percentage=rot_alpha)
        else:
            ##TODO: Solution for now, let's just use SLAHMR optimized results for this.
            ##TODO: Dimension Issue.
            init_rot_world = rot_2D_list[0] 

        ## Merge pose_2D_list ##
        init_pose_latent_world = self.calculate_weighted_averages_pose_latent_2D(pose_latent_2D_list, first_element_weight_percentage=pose_alpha)


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
                cost = compute_cost_l2(pred_smpl_data_array[i], selected_tracks_array[j])
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



    def match_subjects_3d_points_appe(self, pred_smpl_data_list, selected_tracks_smpl_3d_points, init_appe_list_world, selected_appe_list_camera, weight_3d=1.0, weight_appearance=0.0):
        # Convert to numpy arrays
        pred_smpl_data_array = np.array(pred_smpl_data_list)
        selected_tracks_array = np.array(selected_tracks_smpl_3d_points)
        init_appe_array_world = np.array(init_appe_list_world)
        selected_appe_array_camera = np.array(selected_appe_list_camera)

        # Normalize the 3D SMPL coordinates
        ## ! We don't need to normalize the 3D SMPL coordinates because they are already normalized in the SMPL model! ##
        # pred_smpl_data_array = normalize_smpl_features(pred_smpl_data_array)
        # selected_tracks_array = normalize_smpl_features(selected_tracks_array)

        # Normalize the appearance embeddings
        init_appe_array_world = normalize_appearance_embeddings(init_appe_array_world)
        selected_appe_array_camera = normalize_appearance_embeddings(selected_appe_array_camera)

        # Compute the cost matrices
        cost_matrix_combined = np.zeros((len(pred_smpl_data_array), len(selected_tracks_array)))

        for i in range(len(pred_smpl_data_array)):
            for j in range(len(selected_tracks_array)):
                cost_3d = compute_cost_l2(pred_smpl_data_array[i], selected_tracks_array[j])
                # * Ditch Appearance Matching for now # 
                #cost_appe = compute_cost_appearance_euclidean(init_appe_array_world[i], selected_appe_array_camera[j])
                
                # Combine the costs using the provided weights
                #combined_cost = weight_3d * cost_3d + weight_appearance * cost_appe
                combined_cost = weight_3d * cost_3d
                cost_matrix_combined[i, j] = combined_cost

        # Hungarian algorithm for optimal assignment
        row_indices, col_indices = linear_sum_assignment(cost_matrix_combined)

        # Extract matches and unmatched items, same as before
        matched_subjects = [(pred_smpl_data_array[i], selected_tracks_array[j]) for i, j in zip(row_indices, col_indices)]
        unmatched_subjects = [pred_smpl_data_array[i] for i in range(len(pred_smpl_data_array)) if i not in row_indices]

        matched_indices_pairs = list(zip(row_indices, col_indices))
        unmatched_indices_world = [i for i in range(len(pred_smpl_data_array)) if i not in row_indices]
        unmatched_indices_camera = [j for j in range(len(selected_tracks_array)) if j not in col_indices]

        return matched_subjects, unmatched_subjects, matched_indices_pairs, unmatched_indices_world, unmatched_indices_camera



    def initialize_2D_smpl_param_lists(self, init_pose_latent_list, init_betas_list, init_trans_list, init_rot_list, view_init=0):
        # Initialize the 2D lists with the first view's data
        # pose_latent_2D_list = [[init_pose_latent] for init_pose_latent in init_pose_latent_list[view_init]]
        # betas_2D_list = [[init_betas] for init_betas in init_betas_list[view_init]]
        # trans_2D_list = [[init_trans] for init_trans in init_trans_list[view_init]]
        # rot_2D_list = [[init_rot] for init_rot in init_rot_list[view_init]]


        pose_latent_2D_list = []
        betas_2D_list = []
        trans_2D_list = []
        rot_2D_list = []


        return pose_latent_2D_list, betas_2D_list, trans_2D_list, rot_2D_list


    ## TODO: Fixed. Cannot use init_rot_list, instead use obs_data. 
    def select_first_appearance(self, obs_data_list, view_num, t_):
        """
        Input:
        obs_data_list: list of observation data
        view_num: the view number
        t_: the current frame

        Output:
        selected_tracks: list of selected tracks (can turn into numpy)
        """

        # Access the specific view
        view = obs_data_list[view_num]['joints2d']

        # Initialize a list to store the selected tracks
        selected_tracks = []
        selected_tracks_indices = []

        # Initialize a boolean to indicate whether there is a first appearance
        has_first_appearance = False

        # Iterate over the tracks
        for index, track in enumerate(view):
            # Check if the current frame is the first non-zero frame
            if (np.all(track[:t_].to('cpu').numpy() == 0) or (t_== 0) ) and (np.any(track[:t_+1].to('cpu').numpy() != 0) ): ##TODO: Fix 
                selected_tracks.append(track)
                selected_tracks_indices.append(index)
                has_first_appearance = True

        # Convert the list of selected tracks to a numpy array
        # selected_tracks = np.array(selected_tracks)

        return selected_tracks, selected_tracks_indices, has_first_appearance


    def find_last_apperance(self, obs_data):
        """
        Input:
        obs_data: root orientation in the camera frame

        Output:
        t_: last apperance frame 
        """
        for t_ in range(self.seq_len-1):
            if np.any(obs_data[t_].to('cpu').numpy() != 0) and np.all(obs_data[t_+1].to('cpu').numpy() == 0):
                return t_
            elif t_+1 == self.seq_len-1:
                return t_+1
                
    
    def calculate_weighted_averages_pose_latent_2D(self, pose_latent_2D_list, first_element_weight_percentage=50.0):
        """
        Averages the pose latents in a 2D list with weighted averaging on the corresponding pose representation.
        The first latent tensor in each sublist is given a preferential weight based on the specified percentage,
        emphasizing its impact on the weighted average. If a sublist contains only a single pose latent, 
        it is returned as the weighted average without further calculation.

        Parameters:
        pose_latent_2D_list (list of list of torch.Tensor): A list where each element is a sublist containing tensors of pose latent representation.
        first_element_weight_percentage (float): The percentage of the total weight to be assigned to the first element of each sublist.

        Returns:
        torch.Tensor: A tensor containing the weighted averages of the pose latents for each sublist.
        """
        weighted_averages_latent = []

        # Process each sublist
        for latent_list in pose_latent_2D_list:
            if len(latent_list) == 1:
                # If there's only one latent pose, use it as the weighted average directly
                weighted_averages_latent.append(latent_list[0])
            else:
                # Calculate the total weight and the weight of the first element
                total_elements_weight = len(latent_list) - 1  # Total weight of elements except the first
                first_element_weight = total_elements_weight * (first_element_weight_percentage / (100 - first_element_weight_percentage))
                weights = [first_element_weight] + [1] * (len(latent_list) - 1)

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
        self.params.set_batch(batch_size_new)
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

        #* add the multi-view cameras
        cameras_multi_mv = self.params.get_cameras_mv()
        intrins_multi_mv = self.params.intrins_mv
        #? Cannot have cam_Rt_mv as list of tuples ##
        for camera_index in range(1, self.params.num_view):
            res[f"cam_R_{camera_index}"], res[f"cam_t_{camera_index}"], _, _ = cameras_multi_mv[camera_index-1]
            res[f"cam_f_{camera_index}"] = intrins_multi_mv[camera_index-1] #* Here it's actually intrinsics  (cam_f and cam_center)

            
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

    def pred_smpl_adapt(self, trans, root_orient, body_pose, betas, body_model):
        """
        Forward pass of the SMPL model and populates pred_data accordingly with
        joints3d, verts3d, points3d.

        trans : B x T x 3
        root_orient : B x T x 3
        body_pose : B x T x J*3
        betas : B x D
        """

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
            self.params.trans, self.params.root_orient, body_pose, self.params.betas ##Problem! Trans, Root_orient last is nan and 0s. 
        )

        return pred_data
