import os.path as osp
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image


class EpicLocalReader:
    """ Read VISOR Image and Mask  (NOT exactly EPIC-KITCHENS)"""

    IMG_SIZE = (854, 480)

    def __init__(self, data_root):
        self.data_root = Path(data_root)
        self.images_dir = self.data_root/'images'
        self.masks_dir = self.data_root/'masks'
        self.image_format = str(self.images_dir/'%s/%s_frame_%010d.jpg')  # % (vid, vid, frame)
        self.mask_format = str(self.masks_dir/'%s/%s_frame_%010d.png')  # % (vid, vid, frame)

        _frame_to_mappingId = pd.read_csv(
            self.data_root/'visor_meta_infos/frame_to_mappingId.csv')
        self.frame_to_mappingId = {
            (v['vid'], v['frame']): v['mapping_id']
            for _, v in _frame_to_mappingId.iterrows()
            }
        self.mappings = pd.read_csv(
            self.data_root/'visor_meta_infos/unfiltered_color_mappings.csv')

        self.palette = Image.open(
            self.data_root/'visor_meta_infos/00000.png').getpalette()

    def read_image(self, vid, frame) -> np.ndarray:
        img_pil = self.read_image_pil(vid, frame)
        if img_pil is None:
            return None
        return np.asarray(img_pil)

    def read_image_pil(self, vid, frame) -> Image.Image:
        fname = self.image_format % (vid, vid, frame)
        return Image.open(fname).resize(self.IMG_SIZE)

    def read_mask(self, vid, frame, return_mapping=False) -> np.ndarray:
        """
        Caveat: mapping.keys() might contain non-exisitng mask!
            use (mask==mapping[hos_name]).any() to check.

        Returns:
            mask: (H, W, N) np.ndarray
            If return_mapping:
                mapping: {category: int_id} where mask==int_id means category
        """
        mask_path = self.mask_format % (vid, vid, frame)
        if not osp.exists(mask_path):
            return None if not return_mapping else (None, None)
        mask = np.asarray(Image.open(mask_path)).astype(np.uint8)
        if return_mapping:
            mapping_id = self.frame_to_mappingId[ (vid, frame) ]
            df = self.mappings[self.mappings['interpolation'] == mapping_id]
            mapping = {
                v['Object_name']: v['new_index']
                for i, v in df.iterrows()}
            return mask, mapping

        return mask

    def read_mask_pil(self, vid, frame) -> Image.Image:
        m = self.read_mask(vid, frame)
        if m is None:
            return None
        m = Image.fromarray(m)
        m.putpalette(self.palette)
        return m

    def read_blend(self, vid, frame, alpha=0.5) -> Image.Image:
        """
        Returns: list of Image or Image
            (img, mask, overlay)
        """
        m = self.read_mask_pil(vid, frame)
        img_pil = self.read_image_pil(vid, frame)
        img = np.asarray(img_pil)
        m_vals = np.asarray(m)
        m_img_pil = m.convert('RGB')
        m_img = np.asarray(m_img_pil)
        m_img[m_vals == 0] = img[m_vals == 0]
        covered = Image.fromarray(m_img)
        blend = Image.blend(img_pil, covered, alpha)
        return blend
