import mujoco_py
from mujoco_panda import PandaArm
import numpy as np
from numpy import linalg as LA
import time

def jqd():
    return np.dot(p.jacobian(),p.sim.data.qvel[:7].reshape(7,1))

def vel():
    return p.ee_velocity()


def fkj(site_id='ee_site', recompute=True):
    '''
    Compute the forward kinematics for the position and orientation of the
    frame attached to a particular site.
    '''

    if type(site_id) is str:
        site_id = p.sim.model.site_name2id(site_id)

    # Create buffers to store the result.
    jac_shape = (3, p.sim.model.nv)
    jacp = np.zeros(jac_shape, dtype=np.float64).flatten()
    jacr = np.zeros(jac_shape, dtype=np.float64).flatten()

    # Compute Kinematics and
    if recompute:
        p.sim.forward()
    mujoco_py.functions.mj_jacSite(p.sim.model, p.sim.data, jacp, jacr, site_id)

    # Reshape the jacobian matrices and return.
    jacp = jacp.reshape(jac_shape)
    jacr = jacr.reshape(jac_shape)

    return np.vstack([jacp[:, :7], jacr[:, :7]])

def test(l):
    a = {1:1,2:2,3:3}
    try:
        return [a[i] for i in l]
    except KeyError as e:
        print ("error", e)


if __name__ == "__main__":
    # p = PandaArm(render = False)
    p = PandaArm(render = False)
    # p.start_asynchronous_run()
    # input()
    # while True:
    #     p.render()
#     p._sim.data.qpos[:7] =([0.,-0.785,
# 0,
# -2.356,
# 0,
# 1.571,
# 0.785])

    vals = np.zeros(7)
    initial_pose = p._sim.data.qpos.copy()[:7]

    # print (p._model.actuator_names)

    # p._model.actuator_gainprm[:7,0] = 0.

    # print (p.sim.qfrc_bias)

    # period = 0.005
    # elapsed_time_ = 0.0
    p.step()
    p.set_joint_positions(initial_pose)
    p.step()
    # print (elapsed_time_)
    # p._sim.data.ctrl[:7] = initial_pose
    # print ("Done")
    # print (p._model.actuator_gainprm)
    # while True:
        # vals = p.sim.data.qfrc_bias
        # print (vals)
    #     elapsed_time_ += period

    #     delta = 3.14 / 16.0 * (1 - np.cos(3.14 / 5.0 * elapsed_time_)) * 5.5

    #     for j in range(7):
    #         if j == 4:
    #             vals[j] = initial_pose[j] - delta
    #         else:
    #             vals[j] = initial_pose[j] + delta
    #     # print 
        # p._sim.data.qfrc_applied[:] = p.sim.data.qfrc_bias
        # # p.sim.data.ctrl[:] = vals
        # print(p._sim.data.qfrc_applied[:] )
        # p.render()
    # sim.data.ctrl[0] += 0.001
    # sim.data.ctrl[1] += 0.001
    # sim.data.ctrl[2] += 0.001
    # sim.data.ctrl[3] += 0.001
    # sim.data.ctrl[4] += 0.001
    # sim.data.ctrl[5] += 0.001
    # sim.data.ctrl[6] += 0.001
    # sim.data.ctrl[8] += 0.1
    # sim.data.ctrl[7] += 0.1
    # while  True:
    #     p.step()
    # else:
    #     pass


