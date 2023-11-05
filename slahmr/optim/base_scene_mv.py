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

import cv2


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
        init_pose_latent_list, init_betas_list, init_trans_list, init_rot_list = [], [], [], []
        rt_pairs_tensor = []
        for num_view in range(len(obs_data_list)):
            breakpoint()
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
        
            rt_pairs_tensor.append((R_w2c, t_w2c))

            init_pose_latent_list.append(init_pose_latent)
            init_betas_list.append(init_betas)
            init_trans_list.append(init_trans)
            init_rot_list.append(init_rot)


        self.rt_pairs_tensor = rt_pairs_tensor
        # TODO: Obtain world smpl parameters from multi-view smpl parameters, and then use the world smpl parameters to initialize the world smpl model.
        breakpoint()
        init_pose_latent_world, init_betas_world, init_trans_world, init_rot_world, matching_obs_data = self.stitch_world_tracklet(init_pose_latent_list, init_betas_list, 
                                                                                                                                   init_trans_list, init_rot_list, pred_smpl_data_list, obs_data_list)


        self.params.set_param("latent_pose", init_pose_latent_world) ## pose in the world frame
        self.params.set_param("betas", init_betas_world) ## beta in the world frame
        self.params.set_param("trans", init_trans_world) ## root translation in the world frame
        self.params.set_param("root_orient", init_rot_world) ## root orientation in the world frame 


    def stitch_world_tracklet(self, init_pose_latent_list, init_betas_list, init_trans_list, init_rot_list, pred_smpl_data_list, obs_data_list):
        """
        Stitch tracklets from each view into one
        Input
        """        
        print()

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
