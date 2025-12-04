# Linear Algebra functions
from collections import namedtuple
import numpy as np


def toy_two_ray_closest_point():
    # This passes
    A = np.float32([0, 0, 0])
    a = np.float32([1, 0, 0])
    B = np.float32([1, 1, 1])
    b = np.float32([0, 0, -1])
    c = B - A
    ab = np.dot(a, b)
    bc = np.dot(b, c)
    ac = np.dot(a, c)
    aa = np.dot(a, a)
    bb = np.dot(b, b)
    CoefD =  (- ab * bc + ac * bb) / (aa * bb - ab * ab)
    CoefE = (ab * ac - bc * aa) / (aa * bb - ab * ab)
    D = A + a * CoefD
    E = B + b * CoefE
    print("ab", ab)
    print("bc", bc)
    print("ac", ac)
    print("aa", aa)
    print("bb", bb)
    return D, E


def n_ray_closest_point(s, d, FAR_THRESHOLD=10.0) -> np.ndarray:
    """
    Internally, this computes N closest points from one to all other rays,
    resulting (N-1) points on this ray, hence N*(N-1) total points.
    We then take the mean of these points.
    
    https://palitri.com/vault/stuff/maths/Rays%20closest%20point.pdf

    Args:
        s: (N, 3) starting points
        d: (N, 3) directions
        FAR_THRESHOLD: float, if parallel lines, point is infinity.
            Ignore these.
    Returns:
        closest_point: (3) the closest point.
        points_on_ray: (N, N, 3) where diagonal is null.
    """
    retVal = namedtuple('Ret', 'closest_point points_on_rays')
    N = s.shape[0]
    A = s.reshape(N, 1, 3)
    a = d.reshape(N, 1, 3)
    B = s.reshape(1, N, 3)
    b = d.reshape(1, N, 3)
    c = B - A  # (N, N, 3)

    ab = (a*b).sum(-1)
    bc = (b*c).sum(-1)
    ac = (a*c).sum(-1)
    aa = np.tile((a*a).sum(-1), [1, N])
    bb = np.tile((b*b).sum(-1), [N, 1])
    # print("ab", ab[0, 1 ])
    # print("bc", bc[0, 1 ])
    # print("ac", ac[0, 1 ])
    # print("aa", aa[0, 1 ])
    # print("bb", bb[0, 1 ])
    # print("ab.shape", ab.shape)
    # print("bc.shape", bc.shape)
    # print("ac.shape", ac.shape)
    # print("aa.shape", aa.shape)
    # print("bb.shape", bb.shape)
    denominator = aa * bb - ab * ab
    denominator[np.diag_indices_from(denominator)] += 1e-9
    CoefD = (- ab * bc + ac * bb) / denominator
    # CoefE = (ab * ac - bc * aa) / (aa * bb - ab * ab)
    D = A + a * CoefD.reshape(N, N, 1)
    # E = B + b * CoefE.reshape(N, N, 1)
    closest_pts = D[np.tri(N)==0].reshape(N*(N-1)//2, 3)
    closest_pt = closest_pts[
        np.linalg.norm(closest_pts, axis=1) < FAR_THRESHOLD
        ].mean(axis=0)
    points_on_rays = D
    return retVal(closest_point=closest_pt, points_on_rays=points_on_rays)


"""
def n_ray_closest_point(s, d) -> np.ndarray:
    Internally, this computes N closest points from one to all other rays,
    resulting (N-1) points on this ray, hence N*(N-1) total points.
    We then take the mean of these points.
    
    https://palitri.com/vault/stuff/maths/Rays%20closest%20point.pdf

    Args:
        s: (N, 3) starting points
        d: (N, 3) directions
    Returns:
        closest_point: (3) the closest point.
        points_on_ray: (N, N, 3) where diagonal is null.
    retVal = namedtuple('Ret', 'closest_point points_on_rays')
    N = s.shape[0]
    A = s.reshape(N, 1, 3)
    a = d.reshape(N, 1, 3)
    B = s.reshape(1, N, 3)
    b = d.reshape(1, N, 3)
    c = B - A  # (N, N, 3)

    ab = (a*b).sum(-1)
    bc = (b*c).sum(-1)
    ac = (a*c).sum(-1)
    aa = np.tile((a*a).sum(-1), [1, N])
    bb = np.tile((b*b).sum(-1), [N, 1])
    # print("ab", ab[0, 1 ])
    # print("bc", bc[0, 1 ])
    # print("ac", ac[0, 1 ])
    # print("aa", aa[0, 1 ])
    # print("bb", bb[0, 1 ])
    # print("ab.shape", ab.shape)
    # print("bc.shape", bc.shape)
    # print("ac.shape", ac.shape)
    # print("aa.shape", aa.shape)
    # print("bb.shape", bb.shape)
    CoefD = (- ab * bc + ac * bb) / (aa * bb - ab * ab)
    # CoefE = (ab * ac - bc * aa) / (aa * bb - ab * ab)
    D = A + a * CoefD.reshape(N, N, 1)
    # E = B + b * CoefE.reshape(N, N, 1)
    closest_pt = np.mean(D[np.tri(N)==0].reshape(N*(N-1)//2, 3), axis=0)
    points_on_rays = D
    return retVal(closest_point=closest_pt, points_on_rays=points_on_rays)
"""