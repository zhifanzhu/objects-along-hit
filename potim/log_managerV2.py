""" Used by run_v4, run_hand_scale 
Combine original LogManager and EvalManager
"""
import os
import os.path as osp
import hydra
import pandas as pd
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from potim.model.potim_model import POTIM_SC
from potim.utils.open3d.helper import create_global_o3d_render
from moviepy import editor


class LogManagerV2:
    """ 
    LogMangerV2 at per timeline level.
    """

    def __init__(self, 
                 rundir=None,
                 jupyter_rundir=None, 
                 with_o3d_render=False,
                 enable_tensorboard=False):
        if rundir is None:
            self.rundir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        if jupyter_rundir is not None:
            self.rundir = jupyter_rundir

        self.with_o3d_render = with_o3d_render
        if with_o3d_render:
            create_global_o3d_render()
        
        """ Per timeline eval info """
        self.eval_save_dir = osp.join(self.rundir, 'evaluation')
        self.eval_results = []
        os.makedirs(self.eval_save_dir, exist_ok=True)

        """ Per segment info """
        self.potim = None
        self.D = None
        self.timeline_name = None

        """ Full Timeine stitch video cache """
        self._timeline_video_cache = {}

        """ Tensorboard """
        self.enable_tensorboard = enable_tensorboard
        self._step = 0
        if self.enable_tensorboard:
            self.writer = None
    
    def set_potim(self, potim: POTIM_SC, timeline_name, D):
        self.potim = potim
        self.D = D
        self.timeline_name = timeline_name
        self._step = 0

        if self.enable_tensorboard:
            if self.writer is not None:
                self.writer.flush()
                self.writer.close()
                self._step = 0
            self.writer = SummaryWriter(
                log_dir=osp.join(self.rundir, 'logs', timeline_name),
                max_queue=1024)  # make queue big enough to avoid blocking

    ####################
    ### Loss related ###
    def aggregate_loss(self, loss_dict: int, device='cuda'):
        """ This should be deprecated, or only used for debugging.

        # TODO: implement the cached version to avoid disk writing at every step

        Args:
            loss_dict: dict where v is (N, T)
        """
        if not self.enable_tensorboard:
            return self.aggregate_loss_no_tensorboard(loss_dict, device)

        potim = self.potim
        tot_loss = torch.tensor(0.0, device=device)

        # per_init_loss = dict()
        tot_loss_inits = [0.0 for _ in range(potim.num_inits)]
        for k, v in loss_dict.items():
            # print(f"{k}, {v.sum().item():.3f}")
            tot_loss += v.sum()

            # Calculate per-init loss
            _loss = v.sum(-1)
            for _init_i in range(potim.num_inits):
                val = _loss[_init_i].item()
                self.writer.add_scalar(
                    f"init-{_init_i}/{k}", val,
                    global_step=self._step)
                tot_loss_inits[_init_i] += val

        for _init_i in range(potim.num_inits):
            self.writer.add_scalar(
                f"init-{_init_i}/tot_loss", tot_loss_inits[_init_i],
                global_step=self._step)

        return tot_loss

    def aggregate_loss_no_tensorboard(self, loss_dict: int, device='cuda'):
        """
        Args:
            loss_dict: dict where v is (N, T)
        """
        tot_loss = torch.tensor(0.0, device=device)
        for k, v in loss_dict.items():
            tot_loss += v.sum()

        return tot_loss

    def inc_step(self):
        self._step += 1

    def finish_timeline(self):
        if len(self._timeline_video_cache) > 0:
            self.commit_timeline_video_cache()
            self._timeline_video_cache = {}
    
    ##########################
    ### Evaluation related ###
    def add_entry(self, eval_entry):
        self.eval_results.append(eval_entry)

    def save(self):
        df = pd.DataFrame(self.eval_results)
        for timeline_name, df_timeline in df.groupby('timeline_name'):
            df_timeline.to_csv(
                osp.join(self.eval_save_dir, f"{timeline_name}.csv"), index=False)
    
    def check_timeline_exist(self, timeline_name, num_segments):
        """ Check if the timeline and ALL segments exist in result csv 
        used for skipping.
        """
        csv_path = osp.join(self.eval_save_dir, f"{timeline_name}.csv")
        if not osp.exists(csv_path):
            return False
        df = pd.read_csv(csv_path)
        if len(df[df['segi'] == (num_segments-1)]) == 0:
            return False
        return True
    
    ############
    ### I/O ####
    def save_checkpoint(self, suffix: str):
        """
        Save the optimisable parameters of the POTIM model.
        """
        potim = self.potim
        trainable_params = set()
        for k, p in potim.named_parameters():
            if p.requires_grad:
                trainable_params.add(k)

        sd = potim.state_dict()
        sd = {k: v for k, v in sd.items() if k in trainable_params}

        potim_ckpt_path = osp.join(
            self.rundir, f'{self.timeline_name}/potim_{suffix}.ckpt')
        os.makedirs(osp.dirname(potim_ckpt_path), exist_ok=True)
        torch.save(sd, potim_ckpt_path)

    # def load_checkpoint(self, suffix: str):
    #     basename = 
    #     potim_ckpt_path = osp.join(
    #         self.rundir, f'{self.timeline_name}/potim_{suffix}.ckpt')
        
    #     sd = torch.load(potim_ckpt_path)
    #     self.potim.load_state_dict(sd, strict=False)
    
    def save_o2w_poses(self, frames, all_o2w, suffix: str):
        """ Save o2w poses for evaluation
        Args:
            frames: (T,)
            all_o2w: (T, 4, 4)  corresponding to frames
        """
        assert all_o2w.ndim == 3
        o2w_path = osp.join(
            self.rundir, f'{self.timeline_name}/pred_o2w_{suffix}.pt')
        o2w_dict = {
            'frames': frames,
            'o2w': all_o2w.cpu()
        }
        torch.save(o2w_dict, o2w_path)
    
    def save_object_scale(self, scale, suffix: str):
        """ Save o2w poses for evaluation
        Args:
            scale: (1,)
        """
        # This function is deprecated since v6
        scale_o2w_path = osp.join(
            self.rundir, f'{self.timeline_name}/scale_o2w_{suffix}.pt')
        scale_dict = {
            'scale': scale
        }
        torch.save(scale_dict, scale_o2w_path)

    def save_scale_ho(self, scale_hand, scale_obj, suffix: str):
        """ Save the best scale for hand and object
        Args:
            scale_hand: (1,)
            scale_obj: (1,)
        """
        scale_ho_path = osp.join(
            self.rundir, f'{self.timeline_name}/scale_ho_{suffix}.pt')
        scale_dict = {
            'scale_hand': scale_hand, 
            'scale_obj': scale_obj
        }
        torch.save(scale_dict, scale_ho_path)

    ##############################
    ### Video making related #####
    def make_video(self, pose_idx, suffix: str):
        print(f"Making {suffix} video")
        D = self.D
        potim = self.potim
        global_outs = potim.make_compare_video(
            D.global_cam, D.images, pose_idx=pose_idx)
        clip = editor.ImageSequenceClip(global_outs, fps=potim._make_video_fps)
        outpath = osp.join(self.rundir, f"{self.timeline_name}/{suffix}.mp4")
        os.makedirs(osp.dirname(outpath), exist_ok=True)
        clip.write_videofile(outpath, logger=None, verbose=False)

    def make_init_videos(self):
        # print("Making Init videos")
        for _init in range(self.potim.num_inits):
            self.make_video(_init, suffix=f"Init-{_init}")

    def make_final_video(self):
        # print("Making Final Component videos")
        for _init in range(self.potim.num_inits):
            self.make_video(_init, suffix=f"Final-Component-{_init}")

    @torch.no_grad()
    def make_hover_stitch(self, D, pose_idx, segi: int = None, suffix: str = ""):
        """ Stich the hover video with the total video """
        print(f"Making {suffix} video")
        potim = self.potim
        global_outs = potim.make_compare_video(
            D.global_cam, D.images, pose_idx=pose_idx, segi_overwrite=segi)
        if self.with_o3d_render:
            hover_images = potim.make_hover_video(D, pose_idx=pose_idx)
            out_images = []
            for i in range(len(hover_images)):
                out_images.append(self.stitch_images(global_outs[i], hover_images[i]))
        else:
            out_images = global_outs
        clip = editor.ImageSequenceClip(out_images, fps=potim._make_video_fps)
        outpath = osp.join(self.rundir, f"{self.timeline_name}/{suffix}.mp4")
        os.makedirs(osp.dirname(outpath), exist_ok=True)
        clip.write_videofile(outpath, logger=None, verbose=False)

    def cache_hover_stitch(self, D, pose_idx, segi: int):
        """ Cache the Stiching of the hover video with the total video,
        note: will overwrite exsiting. i.e. might get overwritten due to forward-backward pass
        """
        print(f"Caching timeline video cache")
        potim = self.potim
        global_outs = potim.make_compare_video(
            D.global_cam, D.images, pose_idx=pose_idx, segi_overwrite=segi)
        if self.with_o3d_render:
            hover_images = potim.make_hover_video(D, pose_idx=pose_idx)
            out_images = []
            for i in range(len(hover_images)):
                out_images.append(self.stitch_images(global_outs[i], hover_images[i]))
        else:
            out_images = global_outs
        self._timeline_video_cache[segi] = out_images
        return out_images
    
    def commit_timeline_video_cache(self):
        print(f"Making full timeline video")
        potim = self.potim
        out_images = []
        for segi in sorted(self._timeline_video_cache.keys()):
            out_images.extend(self._timeline_video_cache[segi])
        clip = editor.ImageSequenceClip(out_images, fps=potim._make_video_fps)
        outpath = osp.join(self.rundir, f"{self.timeline_name}/FullAfter.mp4")
        os.makedirs(osp.dirname(outpath), exist_ok=True)
        clip.write_videofile(outpath, logger=None, verbose=False)

    @staticmethod
    def stitch_images(image1, image2):
        """
        Stitches two images together vertically, centering them horizontally and filling any extra space with black.

        Parameters:
            image1 (numpy.ndarray): The top image (H1, W1, C).
            image2 (numpy.ndarray): The bottom image (H2, W2, C).

        Returns:
            numpy.ndarray: The stitched image.
        """
        # Determine the dimensions of the final stitched image
        final_width = max(image1.shape[1], image2.shape[1])
        final_height = image1.shape[0] + image2.shape[0]

        # Create a black canvas for the final image
        stitched_image = np.zeros((final_height, final_width, 3), dtype=np.uint8)

        # Place the first image at the top, centered horizontally
        image1_start_x = (final_width - image1.shape[1]) // 2
        stitched_image[:image1.shape[0], image1_start_x:image1_start_x + image1.shape[1]] = image1

        # Place the second image at the bottom, centered horizontally
        image2_start_x = (final_width - image2.shape[1]) // 2
        stitched_image[image1.shape[0]:, image2_start_x:image2_start_x + image2.shape[1]] = image2

        return stitched_image
