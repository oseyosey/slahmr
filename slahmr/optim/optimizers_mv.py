import os
import glob
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from body_model import OP_IGNORE_JOINTS
from geometry.mesh import save_mesh_scenes
from util.logger import Logger, log_cur_stats
from util.tensor import move_to, detach_all
from vis.output import animate_scene, prep_result_vis_mv

from .losses_mv import RootLossMV, SMPLLossMV, MotionLossMV
from .output import save_camera_json


LINE_SEARCH = "strong_wolfe"


"""
Optimization happens in multiple stages
Stage 1: fit root orients and trans of each frame indendpently 
Stage 2: fit SMPL poses and betas of each frame independently
Stage 3: fit poses and roots in same global coordinate frame
"""


class StageOptimizerMV(object):
    def __init__(
        self,
        name,
        model,
        param_names,
        num_views,
        rt_pairs_tensor,
        matching_obs_data,
        lr=1.0,
        lbfgs_max_iter=20,
        save_every=10,
        vis_every=-1,
        save_meshes=True,
        max_chunk_steps=10,
        **kwargs,
    ):
        Logger.log(f"INITIALIZING OPTIMIZER {name} for {param_names}")
        self.name = name
        self.model = model

        self.set_opt_vars(param_names)

        self.optim = torch.optim.LBFGS(
            self.opt_params, max_iter=lbfgs_max_iter, lr=lr, line_search_fn=LINE_SEARCH
        )
        # LBFGS computes losses multiple times per iteration,
        # save a dict mapping iteration to list of stats_dicts
        self.loss_dicts = {}

        self.save_every = save_every
        self.vis_every = vis_every
        self.save_meshes = save_meshes
        self.max_chunk_steps = max_chunk_steps

        self.cur_step = 0

        self.add_chunk = 0
        self.cur_loss = 0
        self.prev_loss = np.inf
        self.last_updated = 0
        self.reached_max = False
        self.reached_max_iter = -1

        ## Garot
        self.num_views = num_views
        self.rt_pairs_tensor = rt_pairs_tensor
        self.matching_obs_data = matching_obs_data

    def set_opt_vars(self, param_names):
        Logger.log("Set param names:")
        Logger.log(param_names)

        self.param_names = param_names
        self.model.params.set_require_grads(self.param_names)
        self.opt_params = [
            getattr(self.model.params, name) for name in self.param_names
        ]

    def forward_pass(self, obs_data):
        raise NotImplementedError

    def load_checkpoint(self, out_dir, device=None):
        if device is None:
            device = torch.device("cpu")

        param_path = os.path.join(out_dir, f"{self.name}_params.pth")
        if os.path.isfile(param_path):
            param_dict = torch.load(param_path, map_location=device)
            self.model.params.load_dict(param_dict)
            Logger.log(f"Params loaded from {param_path}")

        optim_path = os.path.join(out_dir, f"{self.name}_optim.pth")
        if os.path.isfile(optim_path):
            optim_dict = torch.load(optim_path)
            self.optim.load_state_dict(optim_dict["optim"])
            self.cur_step = optim_dict["cur_step"]
            Logger.log(f"Optimizer loaded from {optim_path} at iter {self.cur_step}")

    def save_checkpoint(self, out_dir):
        param_path = os.path.join(out_dir, f"{self.name}_params.pth")
        param_dict = self.model.params.get_dict() #* Paramteres who is being called set_param will be added to Set (including Focal length, Camera, Rotation)
        if "world_scale" in param_dict:
            print("WORLD_SCALE", param_dict["world_scale"].detach().cpu())
        if "floor_plane" in param_dict:
            print("FLOOR PLANE", param_dict["floor_plane"].detach().cpu())
        # if "cam_f" in param_dict:
        #     print("CAM_F", param_dict["cam_f"].detach().cpu())
        torch.save(param_dict, param_path)
        Logger.log(f"Model saved at {param_path}")

        optim_path = os.path.join(out_dir, f"{self.name}_optim.pth")
        torch.save(
            {"optim": self.optim.state_dict(), "cur_step": self.cur_step},
            optim_path,
        )
        Logger.log(f"Optimizer saved at {optim_path}")

    def save_results(self, out_dir, seq_name):
        """
        pred dict will be a dictionary of trajectories.
        each trajectory will have params and lists of trimesh sequences
        """
        os.makedirs(out_dir, exist_ok=True)

        with torch.no_grad():
            pred_dict = self.model.get_optim_result()
        pred_dict = move_to(detach_all(pred_dict), "cpu")

        i = self.cur_step
        for name, results in pred_dict.items():
            # save parameters of trajectory
            out_path = f"{out_dir}/{seq_name}_{i:06d}_{name}_results.npz"
            Logger.log(f"saving params to {out_path}")
            np.savez(out_path, **results)

        # also save the cameras
        with torch.no_grad():
            cam_R, cam_t = self.model.params.get_extrinsics()
            intrins = self.model.params.get_intrinsics()

        save_camera_json(
            f"{out_dir}/{seq_name}_cameras_{self.cur_step:06d}.json",
            cam_R.detach().cpu(),
            cam_t.detach().cpu(),
            intrins.detach().cpu(),
        )

        ## TODO: Also save the mv camera
        with torch.no_grad():
            cam_Rt_mv = self.model.params.get_extrinsics_mv()
            intrins_mv = self.model.params.get_intrinsics_mv()
        
        #save_camera_mv_json(

        # plot losses
        self.plot_losses(out_dir)

    def save_mesh_scenes(self, scene_dict, out_dir):
        # save the meshes
        os.makedirs(out_dir, exist_ok=True)
        i = self.cur_step
        for name, scene in scene_dict.items():
            scene_dir = f"{out_dir}/{i:06d}_{name}"
            mesh_seqs = scene["meshes"]
            Logger.log(f"saving {len(mesh_seqs)} meshes to {scene_dir}")
            save_mesh_scenes(scene_dir, mesh_seqs)

    def vis_result(self, res_dir, obs_data, num_view, vis=None, num_steps=-1):
        if vis is None or self.vis_every < 0:
            return

        # breakpoint()
        # check which results are saved
        seq_name = obs_data["seq_name"][0]
        res_pre = f"{res_dir}/{seq_name}_opt_{self.cur_step:06d}"
        with torch.no_grad():
            pred_dict = self.model.get_optim_result(num_steps=num_steps)
            ##Important: Collect predicted outputs (latent_pose, trans, root_orient, betas, body pose) into dict

        res_dict = detach_all(pred_dict["world"])


        # breakpoint()

        ## * Recreating vis_mask and track_id ##
        batch_world = res_dict['trans'].shape[0]
        length_world = res_dict['trans'].shape[1]

        vis_mask_world = torch.ones((batch_world, length_world)).to(device=obs_data['joints2d'].device) # Render all joints2d in the world
        track_id_world = torch.arange(1,batch_world+1).to(device=obs_data['joints2d'].device)


        scene_dict = move_to(
            prep_result_vis_mv(
                res_dict,
                vis_mask_world,
                track_id_world,
                self.model.body_model, #? I think here model.body_model is already being updated with latest batch size
                num_view,
                floor_plane=obs_data["floor_plane"][0],
            ),
            "cpu",
        )

        # * Consider having a seq name for every view # 
        ## seq_name as paramter in animate_scene.

        animate_scene(vis, scene_dict, res_pre, render_views=["src_cam", "above", "front", "side"])

    def log_losses(self, stats_dict):
        stats_dict = move_to(detach_all(stats_dict), "cpu")
        log_cur_stats(
            stats_dict,
            iter=self.cur_step,
            to_stdout=(self.cur_step % self.save_every == 0),
        )
        for loss_name, loss_val in stats_dict.items():
            loss_dict = self.loss_dicts.get(loss_name, {})
            loss_series = loss_dict.get(self.cur_step, [])
            loss_series.append(loss_val)
            loss_dict[self.cur_step] = loss_series
            self.loss_dicts[loss_name] = loss_dict

    def record_current_losses(self, writer):
        """
        record the mean of current step's loss values in tensorboard
        """
        if len(self.loss_dicts) < 1:
            return

        for loss_name, loss_dict in self.loss_dicts.items():
            loss_mean = np.mean(loss_dict[self.cur_step])
            writer.add_scalar(f"{self.name}/{loss_name}", loss_mean, self.cur_step)

    def plot_losses(self, res_dir):
        """
        plot a box plot for each BFGS iteration
        """
        if len(self.loss_dicts) < 1:
            return
        for loss_name, loss_dict in self.loss_dicts.items():
            # times (list len T)
            # loss vals (list len T of loss value lists)
            times, loss_vals = zip(*loss_dict.items())
            plt.figure()
            plt.boxplot(loss_vals, labels=times, showfliers=False)
            plt.savefig(f"{res_dir}/{loss_name}.png")

    ## GAROT Implementation ##
    def run(self, obs_data_multi, num_iters, out_dir_multi, vis_multi=None, writer=None):
        self.cur_step = 0
        self.loss.cur_step = 0

        # breakpoint()

        out_dir_init = out_dir_multi[0]
        res_dir_init = os.path.join(out_dir_init, self.name) ## Point to the directory 
        os.makedirs(res_dir_init, exist_ok=True)

        obs_data_init = obs_data_multi[0]
        seq_name = "MultiCamera" #obs_data_init["seq_name"][0]
        print("SEQ NAME", seq_name)

        # try to load from checkpoint if exists
        device = obs_data_init["joints2d"].device
        self.load_checkpoint(out_dir_init, device=device)

        if self.cur_step >= num_iters:
            Logger.log(f"Checkpoint at {self.cur_step} >= {num_iters}, skipping")
            return

        Logger.log(f"OPTIMIZING {self.name} FOR {num_iters} ITERATIONS")

        # save initial results and vis
        self.save_results(res_dir_init, seq_name)

        for i in range(self.cur_step, num_iters):
            Logger.log("ITER: %d" % (i))

            if (i + 1) % self.save_every == 0:  # save before
                self.save_checkpoint(out_dir_init)
                self.save_results(res_dir_init, seq_name)
            else:
                self.save_checkpoint(out_dir_init)

            if ((i + 1) % self.vis_every == 0) or (i == 0):  # * render as well as render before init.
                # visualize for every view
                for num_view in range(self.num_views):
                    #breakpoint()
                    self.vis_result(res_dir_init, obs_data_multi[num_view], num_view, vis=vis_multi[num_view])

            self.cur_step = i
            self.loss.cur_step = i

            ## Multi-view optimization
            self.optim_step(obs_data_multi, writer)

            # early termination in case of nans
            if np.isnan(self.cur_loss):
                # we need to backtrack
                self.load_checkpoint(out_dir_init, device=device)
                return

            # termination for the last chunk
            if self.reached_max and self.reached_max_iter < 0:
                self.reached_max_iter = i - 1
            if self.reached_max and i - self.reached_max_iter >= self.max_chunk_steps:
                break

            # termination for middle chunks
            loss_change = self.prev_loss - self.cur_loss
            if self.last_updated == i - 1 and loss_change == 0:
                break
            if (
                (self.cur_loss < 0 and loss_change < 100)
                or (i - self.last_updated >= self.max_chunk_steps)
                or (loss_change < 20 and i - self.last_updated > 5)
            ):
                self.add_chunk = self.add_chunk + 1
                self.last_updated = i
            self.prev_loss = self.cur_loss

        # final save and vis step
        ## TODO: make this multi_view
        self.cur_step = num_iters
        self.save_checkpoint(out_dir_init)
        self.save_results(res_dir_init, seq_name)
        for num_view in range(self.num_views):
            self.vis_result(res_dir_init, obs_data_multi[num_view], num_view, vis=vis_multi[num_view])

    def optim_step(self, obs_data_multi, writer=None):
        def closure():
            # breakpoint()
            self.optim.zero_grad()
            loss, stats_dict, preds = self.forward_pass(obs_data_multi) ## Here it's calling the def forward_pass() function in RootOptimizerMV / SMPLOptimizer / MotionOptimizer
            stats_dict["total"] = loss
            self.log_losses(move_to(detach_all(stats_dict), "cpu"))
            self.cur_loss = stats_dict["total"].detach().cpu().item()
            loss_keys = stats_dict.keys()
            if "motion_prior" in loss_keys and "pose_prior" in loss_keys:
                self.cur_loss = (
                    self.cur_loss
                    - 0.04 * stats_dict["pose_prior"].detach().cpu().item()
                )
            loss.backward()
            return loss

        self.optim.step(closure)
        if writer is not None:
            self.record_current_losses(writer)


class RootOptimizerMV(StageOptimizerMV):
    name = "root_fit_multi_view"
    stage = 0

    def __init__(
        self,
        model,
        all_loss_weights,
        matching_obs_data,
        rt_pairs_tensor,
        num_views,
        use_chamfer=False,
        robust_loss_type="none",
        robust_tuning_const=4.6851,
        joints2d_sigma=100,
        **kwargs,
    ):
        ## optimize global orientation and root translation in the world frame
        param_names = ["trans", "root_orient"]
        #* Optimize also the focal length (and maybe camera translation)
        if model.opt_cams_mv:
            Logger.log(f"{self.name} OPTIMIZING MULTI-VIEW CAMERAS ROTATION AND TRANSLATION")
            for i in range(1, num_views):
                param_names += [f"cam_R_{i}", f"cam_t_{i}"]
        if model.opt_focal_mv:
            Logger.log(f"{self.name} OPTIMIZING MULTI-VIEW CAMERAS FOCAL LENGTH")
            for i in range(1, num_views):
                param_names += [f"cam_f_{i}"]

        super().__init__(self.name, model, param_names, num_views, rt_pairs_tensor, matching_obs_data, **kwargs)

        self.loss = RootLossMV(
            all_loss_weights[self.stage],
            ignore_op_joints=OP_IGNORE_JOINTS,
            joints2d_sigma=joints2d_sigma,
            use_chamfer=use_chamfer,
            robust_loss=robust_loss_type,
            robust_tuning_const=robust_tuning_const,
        )  # Here it's calling the def setup_losses() function

        ## Matching obs data is a ditcionary of dictionary that stores the matching between pred data and obs data. 
            # self.matching_obs_data = matching_obs_data
            #self.rt_pairs_tensor = rt_pairs_tensor # Lists of R,T tuples ( (T, 3, 3), (T, 3) )

    def forward_pass(self, obs_data_multi):
        """
        Takes in observed data, predicts the smpl parameters and returns loss
        """
        # Use current params to go through SMPL and get joints3d, verts3d, points3d
        pred_data = self.model.pred_params_smpl()
        pred_data["cameras"] = self.model.params.get_cameras()
        
        #* Multi-view camera and focal Optimization *#
        pred_data["cameras_multi_mv"] = self.model.params.get_cameras_mv()

        # compute data losses only
        vis_mask_multi = []
        for num_view in range(self.num_views):
            vis_mask = obs_data_multi[num_view]["vis_mask"] >= 0
            vis_mask_multi.append(vis_mask)
        if not vis_mask_multi:
            vis_mask_multi = None

        loss, stats_dict = self.loss(obs_data_multi, pred_data, self.matching_obs_data, self.num_views, valid_mask_multi=vis_mask_multi)
        return loss, stats_dict, pred_data


class SMPLOptimizerMV(StageOptimizerMV):
    name = "smpl_fit_multi_view"
    stage = 0

    def __init__(
        self,
        model,
        all_loss_weights,
        matching_obs_data,
        rt_pairs_tensor,
        num_views,
        use_chamfer=False,
        robust_loss_type="none",
        robust_tuning_const=4.6851,
        joints2d_sigma=100,
        **kwargs,
    ):
        param_names = ["trans", "root_orient", "betas", "latent_pose"]
        if model.opt_scale:
            param_names += ["world_scale"]
        if model.opt_cams_mv:
            Logger.log(f"{self.name} OPTIMIZING MULTI-VIEW CAMERAS ROTATION AND TRANSLATION")
            for i in range(1, num_views):
                param_names += [f"cam_R_{i}", f"cam_t_{i}"]
        if model.opt_focal_mv:
            Logger.log(f"{self.name} OPTIMIZING MULTI-VIEW CAMERAS FOCAL LENGTH")
            for i in range(1, num_views):
                param_names += [f"cam_f_{i}"]

        super().__init__(self.name, model, param_names, num_views, rt_pairs_tensor, matching_obs_data, **kwargs)

        self.loss = SMPLLossMV(
            all_loss_weights[self.stage],
            ignore_op_joints=OP_IGNORE_JOINTS,
            joints2d_sigma=joints2d_sigma,
            use_chamfer=use_chamfer,
            robust_loss=robust_loss_type,
            robust_tuning_const=robust_tuning_const,
        )

    def forward_pass(self, obs_data_multi):
        """
        Takes in observed data, predicts the smpl parameters and returns loss
        """
        # Use current params to go through SMPL and get joints3d, verts3d, points3d
        pred_data = self.model.pred_params_smpl()
        pred_data["cameras"] = self.model.params.get_cameras()
        pred_data["cameras_multi_mv"] = self.model.params.get_cameras_mv()

        pred_data.update(self.model.params.get_vars())


        # compute data losses only
        vis_mask_multi = []
        for num_view in range(self.num_views):
            vis_mask = obs_data_multi[num_view]["vis_mask"] >= 0
            vis_mask_multi.append(vis_mask)
        if not vis_mask_multi:
            vis_mask_multi = None

        loss, stats_dict = self.loss(obs_data_multi, pred_data, self.model.seq_len, self.matching_obs_data, self.num_views, valid_mask_multi=vis_mask_multi)
        return loss, stats_dict, pred_data


class SmoothOptimizerMV(StageOptimizerMV):
    name = "smooth_fit_multi_view"
    stage = 1

    def __init__(
        self,
        model,
        all_loss_weights,
        matching_obs_data,
        rt_pairs_tensor,
        num_views,
        use_chamfer=False,
        robust_loss_type="none",
        robust_tuning_const=4.6851,
        joints2d_sigma=100,
        **kwargs,
    ):
        param_names = ["trans", "root_orient", "betas", "latent_pose"]
        if model.opt_scale:
            param_names += ["world_scale"]
        if model.opt_cams: ##TODO: TEmporary Solution
            param_names += ["cam_f", "delta_cam_R"]


        super().__init__(self.name, model, param_names, num_views, rt_pairs_tensor, matching_obs_data, **kwargs)

        self.loss = SMPLLossMV(
            all_loss_weights[self.stage],
            ignore_op_joints=OP_IGNORE_JOINTS,
            joints2d_sigma=joints2d_sigma,
            use_chamfer=use_chamfer,
            robust_loss=robust_loss_type,
            robust_tuning_const=robust_tuning_const,
        )

    def forward_pass(self, obs_data_multi):
        """
        Takes in observed data, predicts the smpl parameters and returns loss
        """
        # Use current params to go through SMPL and get joints3d, verts3d, points3d
        pred_data = self.model.pred_params_smpl()
        pred_data["cameras"] = self.model.params.get_cameras()
        pred_data["cameras_multi_mv"] = self.model.params.get_cameras_mv()
        pred_data.update(self.model.params.get_vars()) ## ? inserts the output of get_vars(), a dictionary, into the pred_data dictionary

        ## This is originally being commented out
        # camera predictions 
        #         pts_ij, target, weight = self.model.params.get_reprojected_points()
        #         pred_data["bg2d_err"] = (
        #             (weight * (pts_ij - target) ** 2).sum(dim=(-1, -2)).mean()
        #         )

        pred_data["cam_R"], pred_data["cam_t"] = self.model.params.get_extrinsics()

        # compute data losses only
        vis_mask_multi = []
        for num_view in range(self.num_views):
            vis_mask = obs_data_multi[num_view]["vis_mask"] >= 0
            vis_mask_multi.append(vis_mask)
        if not vis_mask_multi:
            vis_mask_multi = None

        loss, stats_dict = self.loss(obs_data_multi, pred_data, self.model.seq_len, self.matching_obs_data, self.num_views, valid_mask_multi=vis_mask_multi)
        return loss, stats_dict, pred_data


class MotionOptimizerMV(StageOptimizerMV):
    name = "motion_fit_multi_view"
    stage = 2

    def __init__(self, model, all_loss_weights, num_views, rt_pairs_tensor, matching_obs_data, opt_cams=False, **kwargs):
        self.opt_cams = opt_cams and model.opt_cams
        param_names = [
            "trans",
            "root_orient",
            "latent_pose",
            "trans_vel",
            "root_orient_vel",
            "joints_vel",
            "betas",
            "latent_motion",
            "floor_plane",
        ]
        if model.opt_scale:
            param_names += ["world_scale"]

        if self.opt_cams:
            Logger.log(f"{self.name} OPTIMIZING CAMERAS")
            param_names += ["delta_cam_R", "cam_f"] ## if opt_cams, ["delta_cam_R", "cam_f"] will be improved.
        
        if model.opt_cams_mv:
            Logger.log(f"{self.name} OPTIMIZING MULTI-VIEW CAMERAS ROTATION AND TRANSLATION")
            for i in range(1, num_views):
                param_names += [f"cam_R_{i}", f"cam_t_{i}"]
        if model.opt_focal_mv:
            Logger.log(f"{self.name} OPTIMIZING MULTI-VIEW CAMERAS FOCAL LENGTH")
            for i in range(1, num_views):
                param_names += [f"cam_f_{i}"]

        #super().__init__(self.name, model, param_names, **kwargs)
        super().__init__(self.name, model, param_names, num_views, rt_pairs_tensor, matching_obs_data, **kwargs) 
        self.set_loss(model, all_loss_weights[self.stage], **kwargs)

    def set_loss(
        self,
        model,
        loss_weights,
        use_chamfer=False,
        robust_loss_type="none",
        robust_tuning_const=4.6851,
        joints2d_sigma=100,
        **kwargs,
    ):
        self.loss = MotionLossMV(
            loss_weights,
            init_motion_prior=model.init_motion_prior,
            ignore_op_joints=OP_IGNORE_JOINTS,
            joints2d_sigma=joints2d_sigma,
            use_chamfer=use_chamfer,
            robust_loss=robust_loss_type,
            robust_tuning_const=robust_tuning_const,
        )

    def get_motion_scale(self):
        return 1.0

    def forward_pass(self, obs_data_list, num_steps=-1):
        p = self.model.params
        param_names = [
            "betas",
            "joints_vel",
            "trans_vel",
            "root_orient_vel",
        ]
        param_dict = p.get_vars(param_names)
        param_dict["floor_plane"] = p.floor_plane[p.floor_idcs]

        preds, world_preds = self.model.rollout_smpl_steps(num_steps)
        preds.update(param_dict)
        world_preds.update(param_dict)

        # track mask is the length of the sequence that contains num_steps of each track
        track_mask = world_preds.get("track_mask", None)

        T = track_mask.shape[1] if track_mask is not None else num_steps
        world_preds["cameras"] = p.get_cameras(np.arange(T))


        ##TODO Update: Track mask is vis_mask but cut in chunks. (Batch_size, T_subset) ##
        track_mask_multi = []
        for num_view in range(self.num_views):
            track_mask_multi.append(track_mask) #! Problem! 
    
        if self.opt_cams: ## ? GAROT will not optimize camera?
            cam_R, cam_t = p.get_extrinsics()
            world_preds["cam_R"], world_preds["cam_t"] = cam_R[:T], cam_t[:T]

        # ? why do we need to slice the obs_data_list? 
        obs_data_list_sliced = []
        for obs_data in obs_data_list:
            obs_data_sliced = slice_dict(obs_data, 0, T)
            obs_data_list_sliced.append(obs_data_sliced)


        motion_scale = self.get_motion_scale()
        loss, stats_dict = self.loss(
            obs_data_list_sliced,
            preds,
            world_preds,
            self.model.seq_len,
            self.matching_obs_data,
            self.num_views,
            valid_mask_multi=track_mask_multi, #* Currently unused
            init_motion_scale=motion_scale,
        )
        return loss, stats_dict, (preds, world_preds)


def slice_dict(d, start, end):
    out = d.copy()
    for k, v in d.items():
        if not isinstance(v, torch.Tensor) or v.ndim < 3 or v.shape[1] < end:
            continue
        out[k] = v[:, start:end]
    return out


class MotionOptimizerChunksMV(MotionOptimizerMV):
    """
    Incrementally optimize sequence in chunks (with all variables)
    :param chunk_size (int) number of frames per chunk
    :param init_steps (int) number of optimization steps to spend on first chunk
    :param chunk_steps (int) number of opt steps to spend on subsequent chunks
    """

    name = "motion_chunks_multi_view"
    stage = 2

    def __init__(self, *args, chunk_size=20, init_steps=20, chunk_steps=10, **kwargs):
        self.chunk_size = chunk_size
        self.init_steps = init_steps
        self.chunk_steps = chunk_steps
        super().__init__(*args, **kwargs)

    @property
    def num_iters(self):
        ## Num Iters are defined by chunk_steps as well
        num_chunks = int(np.ceil(self.model.seq_len / self.chunk_size)) + 1
        return self.init_steps + self.chunk_steps * num_chunks

    @property
    def end_idx(self):
        if self.cur_step < self.init_steps and self.add_chunk == 0:
            return self.chunk_size
        chunk_idx = max(
            (self.cur_step - self.init_steps) // self.chunk_steps + 1, self.add_chunk
        )
        if int(np.ceil(self.model.seq_len / self.chunk_size)) == self.add_chunk:
            self.reached_max = True
        return min(self.chunk_size * (chunk_idx + 1), self.model.seq_len)

    @property
    def start_idx(self):
        return 0

    def get_motion_scale(self):
        num_frames = self.end_idx - self.start_idx
        return max(1.0, float(self.model.seq_len) / num_frames)

    def forward_pass(self, obs_data_list):
        return super().forward_pass(obs_data_list, num_steps=self.end_idx)

    def vis_result(self, res_dir, obs_data, num_view, vis=None, **kwargs):
        print("start, end", self.start_idx, self.end_idx)
        return super().vis_result(res_dir, obs_data, num_view, vis=vis, num_steps=self.end_idx)



class MotionOptimizerFreezeMV(MotionOptimizerMV):
    """
    Only optimizing a subset of parameters
    """

    name = "motion_freeze_multi_view"
    stage = 2

    def __init__(self, model, all_loss_weights, no_contacts=True, **kwargs):
        loss_weights = all_loss_weights[self.stage].copy()

        if no_contacts:
            loss_weights["contact_height"] = 0.0
            loss_weights["contact_vel"] = 0.0

        param_names = ["latent_motion", "betas", "trans", "floor_plane"]
        if model.optim_scale:
            param_names += ["world_scale"]

        StageOptimizer.__init__(self, self.name, model, param_names, **kwargs)
        self.set_loss(model, loss_weights, **kwargs)
