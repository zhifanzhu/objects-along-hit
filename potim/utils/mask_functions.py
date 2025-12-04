import torch
import numpy as np
from skimage.transform import warp, SimilarityTransform


def align_object_mask(mask_ref: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """ Given a ref mask, where 1 indicates foreground,
    translate the whole mask to mask_ref's center and scale.
    Do not rotate, do not scale.

    Returns:
        aligned mask
    """
    def calc_mask_center(mask: torch.Tensor):
        """ return (x, y) i.e. (col, row)"""
        _mask = (mask == 1).nonzero()
        y, x = _mask.float().mean(0).split(1)
        xy = torch.tensor([x, y])
        return xy

    ref_center = calc_mask_center(mask_ref)
    center = calc_mask_center(mask)
    xy_transl = ref_center - center
    tform = SimilarityTransform(
        translation=xy_transl.tolist())  # from src to ref
    itf = tform.inverse  # transform from ref to src
    mask_aligned = torch.from_numpy(warp(mask, itf))
    return mask_aligned, itf


def segment_mask_align(mask: torch.Tensor, images: np.ndarray,
                       st: int, ed: int, ref: int):
    # TODO: handle more than one segment
    out = mask.clone()
    images = images.copy()
    for f in range(st, ed+1):
        out[f], inv_transf = align_object_mask(mask[ref], mask[f])
        _img = warp(images[f], inv_transf)
        images[f] = (_img * 255).astype(np.uint8)
    return out, images


def compute_mask_center(mask: torch.Tensor) -> torch.Tensor:
    """ Note: 
    return the centred xy, i.e. (0, 0) is principal point of the images.
    returns in opencv/open3d format i.e. x-right, y-down, z-inward
    """
    _mask = (mask == 1).nonzero()
    row, col = _mask.float().mean(0).split(1)
    x = col - mask.shape[1] / 2
    y = row - mask.shape[0] / 2
    return torch.tensor([x, y])