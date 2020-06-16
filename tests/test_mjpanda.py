
from mujoco_panda import PandaArm
import numpy as np
from numpy import linalg as LA
import time

if __name__ == "__main__":
    p = PandaArm(render = False)
    p.start_asynchronous_run()
    # p = PandaArm(render = True)
    # input()
    # while True:
    #     p.render()
    p._sim.data.qpos[:7] =([0.,-0.785,
0,
-2.356,
0,
1.571,
0.785])

    vals = np.zeros(7)
    initial_pose = p._sim.data.qpos.copy()[:7]

    # print (p._model.actuator_names)

    p._model.actuator_gainprm[:7,0] = 0.


    period = 0.005
    elapsed_time_ = 0.0
    # print (elapsed_time_)
    # p._sim.data.ctrl[:7] = initial_pose
    # print ("Done")
    # print (p._model.actuator_gainprm)
    # while True:

    #     elapsed_time_ += period

    #     delta = 3.14 / 16.0 * (1 - np.cos(3.14 / 5.0 * elapsed_time_)) * 5.5

    #     for j in range(7):
    #         if j == 4:
    #             vals[j] = initial_pose[j] - delta
    #         else:
    #             vals[j] = initial_pose[j] + delta
    #     # print 
    #     p._sim.data.ctrl[-7:] = vals
    #     p.step()
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


