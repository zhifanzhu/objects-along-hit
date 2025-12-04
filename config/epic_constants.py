IMG_HEIGHT = 720
IMG_WIDTH = 1280

FRANKMOCAP_INPUT_SIZE = 224
# WEAK_CAM_FX = 10

REND_SIZE = 256  # Size of target masks for silhouette loss.

HAND_MASK_KEEP_EXPAND = 0.2

INTERACTION_MAPPING = {
    "default": ["lhand", "rhand"],
}

# Dataset
EPIC_HOA_SIZE = (1920, 1080)
VISOR_HEIGHT = 480
VISOR_WIDTH = 854
VISOR_SIZE = (VISOR_WIDTH, VISOR_HEIGHT)

ALL_CATS = {'bottle', 'bowl', 'plate', 'glass', 'cup', 'mug', 'can', 'pan', 'saucepan'}