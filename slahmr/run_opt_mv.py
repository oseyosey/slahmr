import os
import glob
import json
import subprocess
import pickle

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from data import get_dataset_from_cfg, expand_source_paths, check_cross_view
from pnp.launch_pnp import *

from humor.humor_model import HumorModel
from optim.base_scene import BaseSceneModel
from optim.moving_scene import MovingSceneModel

## GAROT Implementation ##
## TODO: We might want to change the naming to avoid conflict##
from optim.optimizers_mv import (
    RootOptimizerMV,
    SmoothOptimizerMV,
    SMPLOptimizerMV,
    MotionOptimizerMV,
    MotionOptimizerChunksMV,
    CameraOptimizerMV
)

from optim.optimizers import (
    RootOptimizer,
    SmoothOptimizer,
    SMPLOptimizer,
    MotionOptimizer,
    MotionOptimizerChunks,
)


from optim.output import (
    save_track_info,
    save_camera_json,
    save_input_poses,
    save_initial_predictions,
)
from vis.viewer import init_viewer

from util.loaders import (
    load_vposer,
    load_state,
    load_gmm,
    load_smpl_body_model,
    resolve_cfg_paths,
)
from util.logger import Logger
from util.tensor import get_device, move_to, detach_all, to_torch

from run_vis import run_vis

import hydra
from omegaconf import DictConfig, OmegaConf


## Multi-view Implementation ##
from optim.base_scene_mv import BaseSceneModelMV
from optim.moving_scene_mv import MovingSceneModelMV
from pnp.pnp_helpers import get_highest_motion_data

N_STAGES = 3


### GAROT Implementation ###
def run_opt_mv(cfg, dataset_multi, rt_pairs, out_dir_multi, slahmr_data_init, cfg_multi, device, debug=False):
    args = cfg.data

    ## setting up psuedo B and T (B, the number of sequences will change.)
    B_INIT = len(dataset_multi[0])
    T = dataset_multi[0].seq_len

    # check whether the cameras are static
    # if static, cannot optimize scale
    # cfg.model.opt_scale &= not dataset_multi[0].cam_data.is_static
    cfg.model.opt_scale = False # Hardcoded for now
    Logger.log(f"OPT SCALE {cfg.model.opt_scale}")

    # loss weights for all stages
    all_loss_weights = cfg.optim.loss_weights # see optim.yaml config file
    assert all(len(wts) == N_STAGES for wts in all_loss_weights.values())
    stage_loss_weights = [
        {k: wts[i] for k, wts in all_loss_weights.items()} for i in range(N_STAGES)
    ]
    max_loss_weights = {k: max(wts) for k, wts in all_loss_weights.items()}

    # load pose prior models
    cfg = resolve_cfg_paths(cfg)
    paths = cfg.paths
    Logger.log(f"Loading pose prior from {paths.vposer}")
    pose_prior, _ = load_vposer(paths.vposer)
    pose_prior = pose_prior.to(device)


    ## Multi-view Implementation ##
    obs_data_multi = []
    body_model_multi = []
    for view_num in range(cfg.data.multi_view_num):
        dataset = dataset_multi[view_num]
        B = len(dataset_multi[view_num])
        loader = DataLoader(dataset_multi[view_num], batch_size=B, shuffle=False)

        ## obs data is the data gathered from 4D human/PHALP
        obs_data = move_to(next(iter(loader)), device) # move object to device (gpu)
        obs_data_multi.append(obs_data)

        if view_num == 0:
            ## cam data is the R&T from SLAM
            cam_data = move_to(dataset_multi[view_num].get_camera_data(), device)
            print(f"OBS DATA for view {view_num}: ", obs_data.keys())
            print(f"CAM DATA foor view {view_num}: ", cam_data.keys())
            # save cameras
            cam_R, cam_t = dataset.cam_data.cam2world()
            intrins = dataset.cam_data.intrins
            save_camera_json(f"cameras.json", cam_R, cam_t, intrins) # save camera parameters into json

        # body Model requires different batch size for each view 
        Logger.log(f"Loading body model from {paths.smpl}")
        body_model, fit_gender = load_smpl_body_model(paths.smpl, B * T, device=device)
        body_model_multi.append(body_model)

    margs = cfg.model

    ## All poses are in their own INDEPENDENT camera reference frames.
    ## But if images are static, then poses should be in the same camera refernce frames. 
    base_model = BaseSceneModelMV(
                                B_INIT, 
                                T, 
                                body_model_multi, 
                                pose_prior, 
                                fit_gender=fit_gender, 
                                rt_pairs=rt_pairs, 
                                view_nums=cfg.data.multi_view_num,
                                path_body_model=paths.smpl,
                                **margs
                                )

    ## Initialized Multi-view Cameras ##
    rt_pairs_tensor = convert_rt_pairs(cfg, cam_data, rt_pairs, T, device)
    
    ## Initialize base_model with SLAHMR+4D Human results | Multi-view ##
    _, matching_obs_data, B_stitch, body_model_stitch = base_model.initialize(obs_data_multi, cam_data, slahmr_data_init, rt_pairs_tensor)  
    base_model.to(device)

    # save initial results for later visualization
    ## TODO: save initial results under under each view
    ## 这里我们重复的存储了initial prediction的信息到了每个view的output的folder下面，可能会有更好的解决方案
    for view_num, out_dir in enumerate(out_dir_multi):
        args_per_view = cfg_multi[view_num].data
        save_input_poses(dataset, os.path.join(out_dir, "phalp"), args_per_view.seq)
        save_initial_predictions(base_model, os.path.join(out_dir, "init"), args_per_view.seq) # args_per_view.seq is called the name of the sequence e.g. Camera0

    opts = cfg.optim.options
    vis_scale = 0.75
    vis_multi = [] # vis_multi is a list of vis for each view
    for view_num in range(cfg.data.multi_view_num):
        if opts.vis_every > 0:
            vis = init_viewer(
                dataset_multi[view_num].img_size,
                cam_data["intrins"][0],
                vis_scale=vis_scale,
                bg_paths=dataset_multi[view_num].sel_img_paths,
                fps=cfg.fps,
            )
            #breakpoint()
            vis_multi.append(vis)
        else:
            vis_multi = None
    print("OPTIMIZER OPTIONS:", opts)

    writer = SummaryWriter(out_dir_multi[0])

    # * Extracting floor plane from SLAHMR *
    obs_data_multi = extract_ground_plane(cfg, obs_data_multi, slahmr_data_init)

    if debug:
        breakpoint()
    print("RUNNING MULTI-VIEW OPTIMIZATION Stage 1...")
    ##Set up RootOptimizerMV!  ##
    args = cfg.optim.root
    num_iters_root_mv = args.num_iters*cfg.data.multi_view_num ## Chunk size * ITER
    optim = RootOptimizerMV(base_model, stage_loss_weights, matching_obs_data, rt_pairs_tensor, cfg.data.multi_view_num, 
                            opt_cams_mv=args.opt_cams_mv, opt_focal_mv=args.opt_focal_mv, **opts)
    optim.run(obs_data_multi, num_iters_root_mv, out_dir_multi, vis_multi=vis_multi, writer=writer)

    print("RUNNING MULTI-VIEW OPTIMIZATION Stage 1.5: Camera Pose...")
    optim = CameraOptimizerMV(base_model, stage_loss_weights, matching_obs_data, rt_pairs_tensor, cfg.data.multi_view_num, 
                            opt_cams_mv=args.opt_cams_mv, opt_focal_mv=args.opt_focal_mv, **opts)
    optim.run(obs_data_multi, num_iters_root_mv, out_dir_multi, vis_multi=vis_multi, writer=writer)

    if debug:
        breakpoint()
    print("RUNNING MULTI-VIEW OPTIMIZATION Stage 2...")
    args = cfg.optim.smpl
    optim = SMPLOptimizerMV(base_model, stage_loss_weights, matching_obs_data, rt_pairs_tensor, cfg.data.multi_view_num, 
                            opt_cams_mv=args.opt_cams_mv, opt_focal_mv=args.opt_focal_mv, **opts)
    
    num_iters_smooth_mv = args.num_iters
    optim.run(obs_data_multi, num_iters_smooth_mv, out_dir_multi, vis_multi=vis_multi, writer=writer)

    args = cfg.optim.smooth
    optim = SmoothOptimizerMV(
        base_model, stage_loss_weights, matching_obs_data, rt_pairs_tensor, cfg.data.multi_view_num, opt_scale=args.opt_scale, **opts
    )
    optim.run(obs_data_multi, args.num_iters, out_dir_multi, vis_multi=vis_multi, writer=writer) # We might want to change the number of iterations here

    if debug:
        breakpoint()
    
    print("RUNNING MULTI-VIEW OPTIMIZATION Stage 3...")
    # now optimize motion model
    Logger.log(f"Loading motion prior from {paths.humor}")
    motion_prior = HumorModel(**cfg.humor)
    load_state(paths.humor, motion_prior, map_location="cpu") 
    motion_prior.to(device)
    motion_prior.eval()

    Logger.log(f"Loading GMM motion prior from {paths.init_motion_prior}")
    init_motion_prior = load_gmm(paths.init_motion_prior, device=device)

    model = MovingSceneModelMV(
        B_stitch,     ## Here B should be the stitched number of sequences
        T,
        body_model_multi,
        pose_prior,
        motion_prior,
        init_motion_prior,
        fit_gender=fit_gender,
        rt_pairs=rt_pairs,
        rt_pairs_tensor=rt_pairs_tensor,
        view_nums=cfg.data.multi_view_num,
        pairing_info=matching_obs_data,
        body_model_stitch=body_model_stitch,
        path_body_model=paths.smpl,
        **margs,
    ).to(device)


    # initialize motion model with base model predictions
    base_params = base_model.params.get_dict() #* dict_keys(['latent_pose', 'betas', 'root_orient', 'trans']). 
                                               #* These params all lives in the shared world frame.
                                               #* up until now the dimension looks lokay.
                                    
    model.initialize(obs_data_multi, cam_data, rt_pairs_tensor, base_params, cfg.fps) ##GAROT Implementation
    model.to(device)

    if "motion_chunks" in cfg.optim:
        args = cfg.optim.motion_chunks
        optim = MotionOptimizerChunksMV(model, stage_loss_weights, cfg.data.multi_view_num, rt_pairs_tensor, matching_obs_data, **args, **opts)
        #optim = MotionOptimizerMV(model, stage_loss_weights, cfg.data.multi_view_num, rt_pairs_tensor, matching_obs_data, **args, **opts)
        optim.run(obs_data_multi, optim.num_iters, out_dir_multi, vis_multi=vis_multi, writer=writer) # hardcoded

    return 



def convert_rt_pairs(cfg, rt_init, rt_pairs, T, device):
    """
    Convert rt_pairs to list of tensors
    """
    rt_pairs_tensor = []
    for num_view in range(0, cfg.data.multi_view_num):
        # transform into world frame (T, 3, 3), (T, 3)
        if num_view == 0: # world frame
            R_w2c, t_w2c = rt_init["cam_R"], rt_init["cam_t"]
        else: # camera frame, require multi-view rt pairs,
            R_w2c_vec, t_w2c = rt_pairs[num_view]
            t_w2c = torch.from_numpy(t_w2c).to(device)
            R_w2c = cv2.Rodrigues(R_w2c_vec)[0]
            R_w2c =  torch.from_numpy(R_w2c).to(device)

            t_w2c = t_w2c.squeeze().repeat(T, 1).float()
            R_w2c = R_w2c.unsqueeze(0).repeat(T, 1, 1).float()

        rt_pairs_tensor.append((R_w2c, t_w2c))

    return rt_pairs_tensor


def extract_ground_plane(cfg, obs_data_multi, slahmr_data_init):
    """
    Extract ground plane from SLAHMR
    """
    for view_num in range(cfg.data.multi_view_num):     
        import copy    
        floorplane_est = copy.deepcopy(slahmr_data_init["floor_plane"])
        #? not sure if we want to apply floor plane to every view* or just the first view ?#
        obs_data_multi[view_num]["floor_plane"][:] = floorplane_est.expand_as(obs_data_multi[view_num]["floor_plane"])
    return obs_data_multi

### Original SLAHMR Implementation ###
def run_opt(cfg, dataset, out_dir, device):
    args = cfg.data
    B = len(dataset)
    T = dataset.seq_len
    loader = DataLoader(dataset, batch_size=B, shuffle=False)

    ## obs data is the data gathered from 4D human/PHALP
    obs_data = move_to(next(iter(loader)), device) # move object to device (gpu)
    ## cam data is the R&T from SLAM
    cam_data = move_to(dataset.get_camera_data(), device)
    print("OBS DATA", obs_data.keys())
    print("CAM DATA", cam_data.keys())

    # save cameras
    cam_R, cam_t = dataset.cam_data.cam2world()
    intrins = dataset.cam_data.intrins
    save_camera_json(f"cameras.json", cam_R, cam_t, intrins) # save camera parameters into json

    # check whether the cameras are static
    # if static, cannot optimize scale
    cfg.model.opt_scale &= not dataset.cam_data.is_static
    Logger.log(f"OPT SCALE {cfg.model.opt_scale}")

    # loss weights for all stages
    all_loss_weights = cfg.optim.loss_weights # see optim.yaml config file
    assert all(len(wts) == N_STAGES for wts in all_loss_weights.values())
    stage_loss_weights = [
        {k: wts[i] for k, wts in all_loss_weights.items()} for i in range(N_STAGES)
    ]
    max_loss_weights = {k: max(wts) for k, wts in all_loss_weights.items()}

    # load models
    cfg = resolve_cfg_paths(cfg)
    paths = cfg.paths
    Logger.log(f"Loading pose prior from {paths.vposer}")
    pose_prior, _ = load_vposer(paths.vposer)
    pose_prior = pose_prior.to(device)

    Logger.log(f"Loading body model from {paths.smpl}")
    body_model, fit_gender = load_smpl_body_model(paths.smpl, B * T, device=device)

    margs = cfg.model
    ## All poses are in their own INDEPENDENT camera reference frames.
    ## But if images are static, then poses should be in the same camera refernce frames. 
    base_model = BaseSceneModel(
        B, T, body_model, pose_prior, fit_gender=fit_gender, **margs
    )
    base_model.initialize(obs_data, cam_data)
    base_model.to(device)

    # save initial results for later visualization
    save_input_poses(dataset, os.path.join(out_dir, "phalp"), args.seq)
    save_initial_predictions(base_model, os.path.join(out_dir, "init"), args.seq)

    opts = cfg.optim.options
    vis_scale = 0.75
    vis = None
    if opts.vis_every > 0:
        vis = init_viewer(
            dataset.img_size,
            cam_data["intrins"][0],
            vis_scale=vis_scale,
            bg_paths=dataset.sel_img_paths,
            fps=cfg.fps,
        )
    print("OPTIMIZER OPTIONS:", opts)

    writer = SummaryWriter(out_dir)

    optim = RootOptimizer(base_model, stage_loss_weights, **opts)
    optim.run(obs_data, cfg.optim.root.num_iters, out_dir, vis, writer)

    optim = SMPLOptimizer(base_model, stage_loss_weights, **opts)
    optim.run(obs_data, cfg.optim.smpl.num_iters, out_dir, vis, writer)

    args = cfg.optim.smooth
    optim = SmoothOptimizer(
        base_model, stage_loss_weights, opt_scale=args.opt_scale, **opts
    )
    optim.run(obs_data, args.num_iters, out_dir, vis, writer)

    # now optimize motion model
    Logger.log(f"Loading motion prior from {paths.humor}")
    motion_prior = HumorModel(**cfg.humor)
    load_state(paths.humor, motion_prior, map_location="cpu")
    motion_prior.to(device)
    motion_prior.eval()

    Logger.log(f"Loading GMM motion prior from {paths.init_motion_prior}")
    init_motion_prior = load_gmm(paths.init_motion_prior, device=device)

    model = MovingSceneModel(
        B,
        T,
        body_model,
        pose_prior,
        motion_prior,
        init_motion_prior,
        fit_gender=fit_gender,
        **margs,
    ).to(device)

    # initialize motion model with base model predictions
    base_params = base_model.params.get_dict()
    model.initialize(obs_data, cam_data, base_params, cfg.fps)
    model.to(device)

    if "motion_chunks" in cfg.optim:
        args = cfg.optim.motion_chunks
        optim = MotionOptimizerChunks(model, stage_loss_weights, **args, **opts)
        optim.run(obs_data, optim.num_iters, out_dir, vis, writer)

    if "motion_refine" in cfg.optim:
        args = cfg.optim.motion_refine
        optim = MotionOptimizer(model, stage_loss_weights, **args, **opts)
        optim.run(obs_data, args.num_iters, out_dir, vis, writer)


@hydra.main(version_base=None, config_path="confs", config_name="config_mv.yaml")
def main(cfg: DictConfig):
    OmegaConf.register_new_resolver("eval", eval)


    ### First-stage ###
    # 1. Run phalp for all views
    # 2. Obtain GaROT cross-view matching
    # 3. Obtain SLAHMR for the first view 
    #TODO: adaptively select the best view (view 1 might not be the best view)
    # 4. Solve for PnP to obtain camera pose (R, T)

    ## Create a list of OmegaConf Dict for Multi-view ##
    cfg_multi = []
    cfg_multi.append(cfg)
    for num_view in range(1, cfg.data.multi_view_num):
        import copy 
        cfg_mv = copy.deepcopy(cfg)
        cfg_mv.data.seq = f"Camera{num_view}"
        cfg_multi.append(cfg_mv)


    out_dir = os.getcwd() # copies an absolute pathname of the current working directory to the array pointed to by buf, which is of length size.
    print("out_dir", out_dir)

    ## Construct multi-view output directory ##
    out_dir_muli = []
    out_dir_muli.append(out_dir)
    for num_view in range(1, cfg.data.multi_view_num):
        last_segment = out_dir.split('/')[-1]
        out_dir_new = out_dir.replace(last_segment, cfg_multi[num_view].data.name)
        out_dir_muli.append(out_dir_new)


    ## Run PHALP(Tracking) for each view ## 
    dataset_multi = []
    for num_view in range(cfg.data.multi_view_num):     
        if not os.path.exists(f"{out_dir_muli[num_view]}"):
            os.makedirs(f"{out_dir_muli[num_view]}")
        Logger.init(f"{out_dir_muli[num_view]}/opt_log.txt") ## Somehow this is problemtic without building folders (pervious two lines)

        # make sure we get all necessary inputs
        cfg_multi[num_view].data.sources = expand_source_paths(cfg_multi[num_view].data.sources)
        print("SOURCES", cfg_multi[num_view].data.sources)

        """ Example
            SOURCES {'images': '/share/kuleshov/jy928/slahmr/slahmr/output/shelf_dev2/images/Camera0', 
                    'cameras': '/share/kuleshov/jy928/slahmr/slahmr/output/shelf_dev2/slahmr/phalp_out/cameras/Camera0/shot-0', 
                    'tracks': '/share/kuleshov/jy928/slahmr/slahmr/output/shelf_dev2/slahmr/phalp_out/track_preds/Camera0', 
                    'shots': '/share/kuleshov/jy928/slahmr/slahmr/output/shelf_dev2/slahmr/phalp_out/shot_idcs/Camera0.json'}
        """
        dataset = get_dataset_from_cfg(cfg_multi[num_view])  ## return class MultiPeopleDataset
        dataset_multi.append(dataset)
        save_track_info(dataset, out_dir_muli[num_view])

        ## dataset.data_out dictionary output

        phalp_file_path = f"{out_dir_muli[num_view]}/complete_track_data.pkl"
        if not os.path.exists(phalp_file_path):
            with open(phalp_file_path, "wb") as f:
                dataset.load_data()
                pickle.dump(dataset.data_dict, f)
            print("SAVED COMPLETE TRACK INFO")
        else:
            print("COMPLETE TRACK INFO ALREADY EXISTS")
        

        """ Get data for each track
            data_dict = {
                "mask_paths": [],
                "floor_plane": [], ## default ground plane
                "joints2d": [],
                "vis_mask": [],
                "track_interval": [],
                "init_body_pose": [],
                "init_root_orient": [],
                "init_trans": [],
                "init_appe": [] 
            }
        """

    ## Cross view Association ##
    cv_data_path = f"{cfg_multi[0].data.sources.crossview}/cross_view/cross_view_matching_all_frames_data.pickle"
    if not os.path.exists(cv_data_path):
        cv_data_path = check_cross_view(cfg)
        cv_data_path = f"{cfg_multi[0].data.sources.crossview}/cross_view/cross_view_matching_all_frames_data.pickle"
        

    ## Run SLAHMR optimization for view 1 ##
    slahmr_view_num = 0
    device = get_device(0)
    if cfg.run_opt:
        run_opt(cfg_multi[slahmr_view_num], dataset_multi[slahmr_view_num], out_dir_muli[slahmr_view_num], device)

    if cfg.run_vis:
        run_vis(
            cfg_multi[slahmr_view_num], dataset_multi[slahmr_view_num], out_dir_muli[slahmr_view_num], 0, **cfg.get("vis", dict())
        )
    

    ## Run Multi-view PnP to obtain Camera Pose ##
    ## TODO March 2024: Improve the Estimation Module.  
    rt_pairs = run_pnp(cfg, out_dir_muli, out_dir_muli[0], cv_data_path, device, starting_frame=cfg.data.cv_starting_frame, run_pnp=cfg.data.run_pnp, max_iou_threshold=cfg.data.max_iou_threshold)
    print("rt_pairs", rt_pairs)


    ## Setting up paths for obtaining SLAHMR results ##
    slahmr_data_init_path = f"{out_dir_muli[0]}/motion_chunks/" 
    slahmr_data_init_dict = get_highest_motion_data(slahmr_data_init_path)
    slahmr_data_init_dict = {k: torch.from_numpy(v) for k, v in slahmr_data_init_dict.items()}
    slahmr_data_init_dict = move_to(slahmr_data_init_dict, device)

    # breakpoint()

    ### Second-stage: Multi-view Optimization ###
    # 1. Run multi-view optimziation 
    if cfg.run_opt_mv:
        device = get_device(0)
        run_opt_mv(cfg, dataset_multi, rt_pairs, out_dir_muli, slahmr_data_init_dict, cfg_multi, device)

    # if cfg.run_vis:
    #     run_vis(
    #         cfg, dataset, out_dir, 0, **cfg.get("vis", dict())
    #     )


    ### Second-stage ###
    # 1. Run optimization on multi-view


if __name__ == "__main__":
    main()
