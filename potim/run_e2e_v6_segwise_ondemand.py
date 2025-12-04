from typing import Union
from colorama import Fore, Style
import os.path as osp
import numpy as np
import copy
import hydra
from omegaconf import DictConfig, OmegaConf
import tqdm
import torch
from einops import rearrange
from potim.defs.sim3 import Sim3
from potim.model.potim_model import POTIM_SC, SCENE_STATIC, SCENE_DYNAMIC, INHAND
from potim.data.hot3d_segwise_ondemand_dataset import HOT3DOnDemandDataset, _HOT3DSingleVideo
from potim.data.epic_segwise_ondemand_dataset import EPICOnDemandDataset, _EPICSingleVideo
from potim.utils.eval_functions import compute_metrics_wrapper
from potim.utils.propagation_helpers import (
    segi_scheduler, make_o2w_mapping,
    collect_and_transform_pose_inits, pad_to_multiple,
    RecordTracer)
from potim.run_hand_scale import fit_hand_scale_for_v6
from potim.utils.cmd_logger import getLogger, add_hydra_logfile

from potim.log_managerV2 import LogManagerV2
from torch.optim import Adam

"""Usage:
CUDA_VISIBLE_DEVICES=1 HYDRA_FULL_ERROR=1 \
python potim/run_e2e_v6_segwise_ondemand.py  \
    +exp={hot3d, epic}_e2e \
    hydra.run.dir=outputs/2025-03-xxx

where config file lives in config/exp/<dataset>.yaml,
and the base config is in config/potim.yaml which points to hot3d experiment.
"""

log = getLogger(__name__)

@hydra.main(
        version_base="1.2",
        config_path="../config/",
        config_name="potim"
        )  # set to 1.2 so that workdir doesn't change to rundir
def main(cfg: DictConfig) -> None:
    logmanager = LogManagerV2(with_o3d_render=True)
    add_hydra_logfile(log)
    log.info(OmegaConf.to_yaml(cfg))

    if cfg.dataset.version == 'hot3d':
        dataset = HOT3DOnDemandDataset(
            timeline_json_path=cfg.dataset.json_path,
            **cfg.dataset)
    elif cfg.dataset.version == 'epic':
        dataset = EPICOnDemandDataset(
            timeline_json_path=cfg.dataset.json_path,
            **cfg.dataset)

    np.random.seed(0)
    vds_inds = range(max(0, cfg.index_from), min(len(dataset), cfg.index_to))

    if cfg.debug_index is not None:
        vds_inds = [cfg.debug_index]
    if cfg.debug_locate is not None:
        vds_inds = [dataset.locate_index_from_output(cfg.debug_locate)]
    if None in vds_inds:
        raise ValueError("None in vds_inds")

    for i in vds_inds:
        num_static = sum(1 for seg in dataset.timelines[i]['segments'] if seg['ref'] == SCENE_STATIC)
        num_inhand = sum(1 for seg in dataset.timelines[i]['segments'] if seg['ref'] == INHAND)
        num_dynamic = sum(1 for seg in dataset.timelines[i]['segments'] if seg['ref'] == SCENE_DYNAMIC)
        if cfg.skip_existing and \
            logmanager.check_timeline_exist(
                dataset.timelines[i]['timeline_name'], dataset.get_num_segments(i)):
            log.info(f"Skipping {i} as it already exists")
            continue
        log.info(
            Fore.GREEN + f"Running at index [{i}]: \
            Static: {num_static}, InHand: {num_inhand}, Dynamic: {num_dynamic} \
            timeline_name: {dataset.timelines[i]['timeline_name']}" + Style.RESET_ALL)
        vds = dataset[i]
        if cfg.only_cat and cfg.only_cat == vds.tl['cat']:
            log.info(f"Skipping {i} for {cfg.only_cat=}")

        log.info(Fore.WHITE + 'COP Starts' + Style.RESET_ALL)
        r = fit_potim_sequential(vds, cfg, logmanager)

        logmanager.save()


def fit_potim_sequential(vds: Union[_HOT3DSingleVideo, _EPICSingleVideo],
                         cfg: dict,
                         logmanager: LogManagerV2):
    # variable to keep track of the previous pose index. Otherwise, we try to find the previous pose when there is a segment with 0 samples.
    all_prev_o2w, prev_scale_hand, prev_scale_obj = None, None, None
    segi_list = segi_scheduler(vds.segments, cfg.segi_strategy)
    record_tracer = RecordTracer()
    print(Fore.GREEN, "Segi_list: ", 
        [ (segi, vds.segments[segi]['ref']) for segi in segi_list], Style.RESET_ALL)
    for work_id, segi in enumerate(segi_list):
        if cfg.debug_segi is not None and segi != cfg.debug_segi:
            continue
        if cfg.load_from_segi is not None:
            if segi != cfg.load_from_segi + 1:
                continue
            else:
                timeline_name = vds.single_segment_timelines[segi]['timeline_name']
                path = osp.join(logmanager.rundir, timeline_name, f"pred_o2w_{cfg.load_from_segi}.pt")
                pred_o2w = torch.load(path)
                all_prev_o2w = make_o2w_mapping(pred_o2w)
                log.info(f"Loading from segment {cfg.load_from_segi}")
        if cfg.which_only != 'fullpassing' and (_ref := vds.get_ref(segi)) != cfg.which_only:
            log.info(f"Skipping segment {segi=} with {_ref=} != {cfg.which_only}")
            continue
        D = vds[segi]
        if D is None:
            log.info("Skipping segment with 0 samples.")
            continue
        D.segi = segi
        D.work_id = work_id

        # The Passing Control
        D = collect_and_transform_pose_inits(
            all_prev_o2w, prev_scale_hand, prev_scale_obj, D=D, which_only=cfg.which_only)

        log.info(Fore.CYAN + f"Processing Seg {segi} ref={D.segments[0].ref}" + Style.RESET_ALL)
        all_prev_o2w, prev_scale_hand, prev_scale_obj = fit_potim_piece(D, cfg, logmanager, record_tracer)

        if D.segments[0].ref == SCENE_STATIC and D.dataset_name == 'epic':
            # Need to fit_hand_scale post static
            best_scale_hand = fit_hand_scale_for_v6(
                all_prev_o2w, prev_scale_hand, prev_scale_obj, vds)
            prev_scale_hand = best_scale_hand

        logmanager.save()

    logmanager.finish_timeline()
    return True

def fit_potim_piece(D, cfg, logmanager: LogManagerV2, record_tracer: RecordTracer) -> dict:
    """
    We gather metrics for each initialisation,
    get the arg-best index, and:
        - plot the best video
        - save all but sort by oious

    Returns:
        prev_all_o2w: dict mapping from frames to best o2w poses
    """
    if record_tracer.is_indep_init_completed(D.segi):
        Np_offset = D.num_inits - 1
        num_inits_run = 1  # num inits to run
    else:
        Np_offset = 0
        num_inits_run = D.num_inits  # all the inits
    Np = min(cfg.optim_mv.num_inits_parallel, num_inits_run)  # num parallel inits
    np_epochs = np.ceil(num_inits_run / Np).astype(int)
    D.Np = Np
    D = pad_to_multiple(D, Np)

    # Common things can be done once
    potim = POTIM_SC(
        D.meta_samples,
        ndof=cfg.potim.ndof,
        num_inits=Np,
        update_scale=cfg.optim_mv.optimize_obj_scale)
    potim.to('cpu')
    potim.register_obj_buffer(D.obj.verts, D.obj.faces)
    potim.register_camera(D.roi_camintr)
    potim.register_obj_target(D.obj.mask_patch)
    potim.register_w2cs(D.w2cs)
    potim.to('cuda')
    potim.set_video_frames('all')
    potim.set_display_segi(D.segi)
    potim.image_patches = D.image_patches  # for saving optim video

    logmanager.set_potim(potim, D.timeline_name, D)

    for e in range(np_epochs):
        D.Np_start = e * Np + Np_offset
        D.Np_end = min((e+1)*Np + Np_offset, D.num_inits)
        all_o2w, eval_entry, eval_entry_before, sd_before_optim, sd_after_optim \
            = fit_potim_piece_epoch(D, cfg, potim, logmanager)

        scale_obj = potim.scale_obj_getter().detach().cpu().view(-1)  # (Np,)
        scale_hand = D.scale_hand  # (Np,)
        blob = (
            eval_entry, eval_entry_before, all_o2w, sd_before_optim, sd_after_optim,
            scale_hand, scale_obj, D.work_id, D.init_types, D.Np_start, D.Np_end)
        record_tracer.update_record(D.segi, blob, logmanager)
    best_record = record_tracer.fetch_best(D.segi)
    log.info(f"{best_record.eval_entry_before=}")
    log.info(f"{best_record.entry=}")

    best_sd_before_optim = best_record.sd_before_optim
    best_sd_after_optim = best_record.sd_after_optim
    best_all_o2w = best_record.all_o2w
    best_init_idx = best_record.init_idx
    best_scale_hand = best_record.scale_hand
    best_scale_obj = best_record.scale_obj

    # TODO: I still couldn't make num_parallel_inits=2 work, due to best_sd has bsize 2 while potim might only have 1
    if cfg.make_video:
        best_pose_idx = best_init_idx % Np
        name_fmt = f"Work[{D.work_id:02d}]_seg[{D.segi:02d}]_%s"
        logmanager.potim.load_state_dict(best_sd_before_optim)
        logmanager.make_hover_stitch(D=D, pose_idx=best_pose_idx, segi=D.segi, suffix=name_fmt % "Before")
        logmanager.potim.load_state_dict(best_sd_after_optim)
        logmanager.make_hover_stitch(D=D, pose_idx=best_pose_idx, segi=D.segi, suffix=name_fmt % "After")
    if cfg.make_timeline_video:
        # Only do After
        logmanager.potim.load_state_dict(best_sd_after_optim)
        logmanager.cache_hover_stitch(D=D, pose_idx=best_init_idx%Np, segi=D.segi)

    # only save the best one
    CUR = 0
    suffix = f"{D.segi}_{D.work_id}"
    logmanager.potim.load_state_dict(best_sd_after_optim)
    logmanager.save_checkpoint(suffix=suffix)
    logmanager.save_o2w_poses(
        frames=D.meta_samples.frames_per_seg[CUR],
        all_o2w=best_all_o2w, suffix=suffix)
    prev_all_o2w = make_o2w_mapping(
        dict(frames=D.meta_samples.frames_per_seg[CUR], o2w=best_all_o2w))
    if D.dataset_name == 'epic':
        logmanager.save_scale_ho(best_scale_hand, best_scale_obj, suffix=suffix)

    return prev_all_o2w, best_scale_hand, best_scale_obj


def fit_potim_piece_epoch(D, cfg, potim: POTIM_SC, logmanager):
    """
    Returns:
        all_prev_o2w: (Np, T, 4, 4)
        eval_entry: dict
            values can be
                - str
                - int
                - tensor of (Np,)
    """
    Np_start = D.Np_start
    Np_end = D.Np_end
    cur_segi = D.segi

    segments = D.meta_samples.segments
    CUR = 0  # due to on-demand dataset, we only have one segment

    potim.init_hand(D, Np_start, Np_end)
    potim.init_scale_obj(D.scale_obj, Np_start, Np_end)
    """ Init the current segment """
    seg = segments[CUR]
    print(f"Optimise {seg.ref} init_ind={Np_start}-{Np_end}")
    if seg.ref == SCENE_STATIC:
        T_o2w = D.static_inits_o2w_list[CUR].to_matrix()  # (N, 1, 4, 4)
        T_o2w = T_o2w[Np_start:Np_end, ...]  # (Np, 1, 4, 4)
        potim.M[CUR].init_obj_transform(Sim3.from_matrix(T_o2w), in_coord='o2w')
    elif seg.ref in {SCENE_DYNAMIC, INHAND}:
        T_o2h = D.init_o2h_list[CUR].to(potim.device) # (N, 4, 4)
        T_o2h = T_o2h[Np_start:Np_end, ...]  # (Np, 4, 4)
        T_o2h = rearrange(T_o2h, 'n i j -> n 1 i j')  # (Np, 1, 4, 4)
        T_o2h = Sim3.from_matrix(T_o2h)  # (Np, 1, 4, 4)
        potim.M[CUR].init_obj_transform(T_o2h, in_coord='o2h')
    else:
        raise ValueError(f"Unknown ref: {seg.ref}")
    """ Init current segment DONE """
    potim.to('cuda')

    sd_before_optim = copy.deepcopy(potim.state_dict())
    abs_inds = torch.arange(seg.st, seg.ed+1)
    eval_entry_before, _ = compute_metrics_wrapper(seg, potim, cfg, D, CUR, cur_segi)

    optimise_phase(abs_inds, cfg, potim, logmanager)

    sd_after_optim = copy.deepcopy(potim.state_dict())
    eval_entry, all_o2w = compute_metrics_wrapper(
        seg, potim, cfg, D, CUR, cur_segi)

    return all_o2w, eval_entry, eval_entry_before, sd_before_optim, sd_after_optim

def optimise_phase(abs_inds, cfg, potim: POTIM_SC, logmanager,
                   save_optim_video=False):
    optimizer = Adam(
        params=potim.parameters(),
        lr=cfg.optim_mv.lr,
    )
    num_iterations = cfg.optim_mv.num_iters
    n_samples_per_run = cfg.optim_mv.n_samples_per_run
    _, num_runs = get_run_inds(abs_inds, 0, n_samples_per_run)
    pbar = tqdm.tqdm(range(num_iterations))
    if save_optim_video:
        optim_frames = []
    for _iter in range(num_iterations):
        for _nrun in range(num_runs):
            inds, _ = get_run_inds(abs_inds, _nrun, n_samples_per_run)
            ret = potim.train_loss(inds, cfg.optim_mv, debug_check_nan=False)
            loss_dict = ret.loss_dict
            # loss = logmanager.aggregate_loss(loss_dict)  # with tensorboard but too slow!
            loss = logmanager.aggregate_loss_no_tensorboard(loss_dict)
            optimizer.zero_grad()
            loss.backward()
            logmanager.inc_step()
            if save_optim_video:
                optim_frames.append((potim.render_grid_np(pose_idx=0, with_hand=True) * 255).astype(np.uint8))

            for name, param in potim.named_parameters():
                if param.grad is None or not param.requires_grad:
                    continue
                if torch.isnan(param.grad).any():
                    print(f"WARNING: cleaning grad for {name}")
                    param.grad[torch.isnan(param.grad)] = 0
                    # print(name, torch.isnan(param.grad))
                    # import pudb; pudb.set_trace()
            optimizer.step()
        pbar.update(1)
        pbar.set_description(f'Iter {_iter}, Loss = {loss.item():.3f}')
    if save_optim_video:
        from moviepy.editor import ImageSequenceClip
        clip = ImageSequenceClip(optim_frames, fps=10)
        clip.write_videofile(f'optim_video.mp4', codec='libx264')


def get_run_inds(abs_inds, nth, n_samples_per_run):
    num_samples = len(abs_inds)
    num_per_run = min(num_samples, n_samples_per_run)
    n_runs = (num_samples + num_per_run - 1) // num_per_run
    sel = torch.arange(num_per_run) * n_runs + nth
    sel = sel[sel < len(abs_inds)]
    inds = abs_inds[sel]
    return inds, n_runs


if __name__ == '__main__':
    main()
