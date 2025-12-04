import numpy as np
import torch
from pytorch3d.transforms import (
    matrix_to_quaternion, quaternion_to_matrix,
    rotation_6d_to_matrix, matrix_to_rotation_6d,
)


def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])


def slerp(q1, q2, times) -> np.ndarray:
    # Ensure that the input quaternions are unit quaternions
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)

    # Calculate the dot product of the quaternions
    dot = np.dot(q1, q2)

    # If the dot product is negative, flip the second quaternion
    if dot < 0:
        q2 = -q2
        dot = -dot

    # Set the interpolation parameters
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    # t = np.linspace(0, 1, N+2)[1:-1]
    t = times

    # Interpolate between the quaternions
    w1 = np.sin((1 - t) * theta) / sin_theta
    w2 = np.sin(t * theta) / sin_theta
    q_interp = np.outer(w1, q1) + np.outer(w2, q2)

    # Normalize the interpolated quaternions and return them as a list of arrays
    # result = [q1]
    # result.extend([q / np.linalg.norm(q) for q in q_interp])
    result = np.asarray([q / np.linalg.norm(q) for q in q_interp])
    return result


def interpolate_individual_Rt(pose1: torch.Tensor,
                              pose2: torch.Tensor,
                              times: torch.Tensor) -> torch.Tensor:
    """
    Similar to https://docs.ros.org/en/melodic/api/gtsam/html/classgtsam_1_1Pose3.html#aa7ada6370ca971fe8170213bacc35d77
    perform slerp on rotation and linear interpolation on translation.

    Args:
        pose1: (4, 4)
        pose2: (4, 4)
        times: (N,)

    Returns:
        interp_poses: (N, 4, 4)
    """
    R1 = pose1[:3, :3]
    R2 = pose2[:3, :3]
    q1 = matrix_to_quaternion(R1)
    q2 = matrix_to_quaternion(R2)
    # key_rots = Rotation.from_matrix(torch.stack([R1, R2]))
    # slerp_func = Slerp([0, 1], key_rots)
    # interp_rots = slerp_func(times)
    # interp_rots = torch.from_numpy(interp_rots.as_matrix())  # (N, 3, 3)
    interp_quats = torch.from_numpy(slerp(q1.numpy(), q2.numpy(), times))
    interp_rots = quaternion_to_matrix(interp_quats)

    def interp_points(loc1, loc2, times):
        loc1 = loc1.view(1, 3)
        loc2 = loc2.view(1, 3)
        times = torch.tensor(times).view(-1, 1)
        return loc1 + times * (loc2 - loc1)
    interp_transl = interp_points(pose1[:3, 3], pose2[:3, 3], times)

    interp_poses = torch.eye(
        4, device=pose1.device, dtype=pose1.dtype
        ).unsqueeze(0).repeat(len(times), 1, 1)
    interp_poses[:, :3, :3] = interp_rots
    interp_poses[:, :3, 3] = interp_transl
    return interp_poses


""" Average quaternions
# https://math.stackexchange.com/questions/61146/averaging-quaternions
"""
def avg_rot6d_approx(rot6d: torch.Tensor, weights=None) -> torch.Tensor:
    """
    Args:
        rot6d: (N, 6), if flat=True
    Returns:
        rot6dAvg: (6,)
    """
    quats = matrix_to_quaternion(rotation_6d_to_matrix(rot6d))
    q_avg = avg_quaternions_approx(quats, weights)
    return matrix_to_rotation_6d(quaternion_to_matrix(q_avg))

def avg_matrix_approx(matrices: torch.Tensor, weights=None) -> torch.Tensor:
    """
    Args:
        matrices: (N, 3, 3) apply to col-vec
    Returns:
        matrix: (3, 3)
    """
    quats = matrix_to_quaternion(matrices)
    q_avg = avg_quaternions_approx(quats, weights)
    return quaternion_to_matrix(q_avg)


def avg_quaternions_approx(quats: torch.Tensor, weights=None) -> torch.Tensor:
    """
    Args:
        quats: (N, 4)
    Returns:
        qAvg: (4,)
    """
    if weights is not None and len(quats) != len(weights):
        raise ValueError("Args are of different length")
    if weights is None:
        weights = torch.ones_like(quats[:, 0])
    qAvg = torch.zeros_like(quats[0])
    for i, q in enumerate(quats):
        # Correct for double cover, by ensuring that dot product
        # of quats[i] and quats[0] is positive
        if i > 0 and torch.dot(quats[i], quats[0]) < 0.0:
            weights[i] = -weights[i]
        qAvg += weights[i] * q
    return qAvg / torch.norm(qAvg)


def avg_rot6d_eigen(rot6d: torch.Tensor, weights=None) -> torch.Tensor:
    """
    Args:
        rot6d:
            - (N, 6), if flat=True
    Returns:
        rot6dAvg: (6,)
    """
    quats = matrix_to_quaternion(rotation_6d_to_matrix(rot6d))
    q_avg = avg_quaternions_eigen(quats)
    return matrix_to_rotation_6d(quaternion_to_matrix(q_avg))


def avg_quaternions_eigen(quats: torch.Tensor, weights=None) -> torch.Tensor:
    # Why flipped?
    raise NotImplementedError
    if weights is not None and len(quats) != len(weights):
        raise ValueError("Args are of different length")
    if weights is None:
        weights = torch.ones_like(quats[:, 0])
    accum = torch.zeros((4, 4), device=quats.device)
    for i, q in enumerate(quats):
        qOuterProdWeighted = torch.outer(q, q) * weights[i]
        accum += qOuterProdWeighted
    _, eigVecs = torch.symeig(accum, eigenvectors=True)
    return eigVecs[:, 0]


""" 3d vector calculation """
def make_basis(w: torch.Tensor) -> torch.Tensor:
    """ Make basis from a column vector
    Args:
        w: (3), this will be the last column, i.e. Z axis
    Returns:
        basis: (3, 3) as [u, v, w]
    """
    w = w / torch.norm(w, dim=-1, keepdim=True)
    u = find_perpendicular(w)
    u = u / torch.norm(u, dim=-1, keepdim=True)
    v = torch.cross(w, u)
    basis = torch.zeros((3, 3), device=w.device, dtype=w.dtype)
    basis[:, 0] = u
    basis[:, 1] = v
    basis[:, 2] = w
    return basis

def find_perpendicular(v: torch.Tensor) -> torch.Tensor:
    """ A branch-free algorithm to find a perpendicular vector to v
    that guarantees non-zero & non-parallel
    u = [ copysign(z, x), copysign(z, y), - copysign(|x|+|y|, z) ]
    src: https://math.stackexchange.com/a/4112622/838554

    Args:
        v: (3)
    Returns:
        u: (3), note, result is unnormalised
    """
    u = torch.zeros_like(v)
    u[0] = torch.copysign(v[2], v[0])
    u[1] = torch.copysign(v[2], v[1])
    u[2] = - torch.copysign(v[0].abs() + v[1].abs(), v[2])
    return u
