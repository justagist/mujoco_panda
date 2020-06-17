import mujoco_py
import numpy as np
from scipy.spatial.transform import Rotation as R

def euler2quat(euler, degrees=True, flip=False):
    r = R.from_euler('xyz', euler, degrees=degrees)
    q = r.as_quat()
    if flip:
        return np.hstack([q[-1], q[:3]])
    return q

def quat2euler(quat, degrees=False, flip=False):
    if flip:
        quat = np.hstack([quat[1:4], quat[0]])
    r = R.from_quat(quat)
    return r.as_euler('xyz', degrees=degrees)

def euler2mat(euler, degrees=True):
    r = R.from_euler('xyz', euler, degrees=degrees)
    return r.as_matrix()

def quat2mat(quat, flip=False):
    if flip:
        quat = np.hstack([quat[1:4], quat[0]])
    r = R.from_quat(quat)
    return r.as_matrix()


# def quatdiff(quat_curr, quat_des):
#     #quat_curr*quat_des.conjugate()
#     qdiff = 2*(quat_curr*quat_des.conjugate()).log()
#     qdiff_array = quaternion.as_float_array(qdiff)
#     return np.fmod(qdiff_array[1:],np.pi)

identity_quat = np.array([1., 0., 0., 0.])


def mat2Quat(mat):
    '''
    Convenience function for mju_mat2Quat.
    '''
    res = np.zeros(4)
    mujoco_py.functions.mju_mat2Quat(res, mat.flatten())
    return res


def quat2Mat(quat):
    '''
    Convenience function for mju_quat2Mat.
    '''
    res = np.zeros(9)
    mujoco_py.functions.mju_quat2Mat(res, quat)
    res = res.reshape(3, 3)
    return res


def quat2Vel(quat):
    '''
    Convenience function for mju_quat2Vel.
    '''
    res = np.zeros(3)
    mujoco_py.functions.mju_quat2Vel(res, quat, 1.)
    return res


def axisAngle2Quat(axis, angle):
    '''
    Convenience function for mju_quat2Vel.
    '''
    res = np.zeros(4)
    mujoco_py.functions.mju_axisAngle2Quat(res, axis, angle)
    return res


def subQuat(qb, qa):
    '''
    Convenience function for mju_subQuat.
    '''
    # Allocate memory
    qa_t = np.zeros(4)
    q_diff = np.zeros(4)
    res = np.zeros(3)

    # Compute the subtraction
    mujoco_py.functions.mju_negQuat(qa_t, qa)
    mujoco_py.functions.mju_mulQuat(q_diff, qb, qa_t)
    mujoco_py.functions.mju_quat2Vel(res, q_diff, 1.)

    #   Mujoco 1.50 doesn't support the subQuat function. Uncomment this when
    #   mujoco_py upgrades to Mujoco 2.0
    # res = np.zeros(3)
    # mujoco_py.functions.mju_subQuat(res, qa, qb)
    return res


def mulQuat(qa, qb):
    res = np.zeros(4)
    mujoco_py.functions.mju_mulQuat(res, qa, qb)
    return res


def random_quat():
    q = np.random.random(4)
    q = q/np.linalg.norm(q)
    return q


def quatIntegrate(q, v, dt=1.):
    res = q.copy()
    mujoco_py.functions.mju_quatIntegrate(res, v, 1.)
    return res


def quatAdd(q1, v):
    qv = quatIntegrate(identity_quat, v)
    res = mulQuat(qv, q1)
    return res


def rotVecQuat(v, q):
    res = np.zeros(3)
    mujoco_py.functions.mju_rotVecQuat(res, v, q)
    return res
