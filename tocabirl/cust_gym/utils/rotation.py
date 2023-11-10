from gym.envs.robotics.rotations import *

_FLOAT_EPS = np.finfo(np.float64).eps
_EPS4 = _FLOAT_EPS * 4.0


def quat2fixedXYZ(quat):
    """Convert Quaternion to Euler Angles.  See rotation.py for notes"""
    return mat2fixedXYZ(quat2mat(quat))


def mat2fixedXYZ(mat):
    """Convert Rotation Matrix to Fixed XYZ Angles.  See rotation.py for notes"""
    mat = np.asarray(mat, dtype=np.float64)
    assert mat.shape[-2:] == (3, 3), "Invalid shape matrix {}".format(mat)

    cy = np.sqrt(mat[..., 0, 0] * mat[..., 0, 0] + mat[..., 1, 0] * mat[..., 1, 0])
    condition = cy > _EPS4
    fixed_angle = np.empty(mat.shape[:-1], dtype=np.float64)
    fixed_angle[..., 2] = np.where(
        condition,
        np.arctan2(mat[..., 1, 0], mat[..., 0, 0]),
        np.arctan2(-mat[..., 0, 1], mat[..., 1, 1]),
    )
    fixed_angle[..., 1] = np.where(
        condition, np.arctan2(-mat[..., 2, 0], cy), np.arctan2(-mat[..., 2, 0], cy)
    )
    fixed_angle[..., 0] = np.where(
        condition, np.arctan2(mat[..., 2, 1], mat[..., 2, 2]), 0.0
    )
    return fixed_angle
