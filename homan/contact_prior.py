from typing import NamedTuple, List
from libzhifan import io
from hydra.utils import to_absolute_path

class ContactRegion(NamedTuple):
    """
    In total 8 prior regions = 
        5 fingers ordered from {thumb, index, middle, ring, pinky} +
        3 palms parts ordered from {below-thumb, finger-root, edge}
    
    verts: contains indices to hand vertices
    """
    verts: List[List]
    faces: List[List]


def get_contact_regions(path='weights/contact_regions.json'):
    contact_regions = io.read_json(to_absolute_path(path))
    return ContactRegion(**contact_regions)
