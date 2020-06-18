import mujoco_py
from mujoco_panda import PandaArm
import numpy as np
from numpy import linalg as LA
import os
import time
import threading
import quaternion
from mujoco_panda.utils.viewer_utils import render_frame
from mujoco_panda.utils.tf import quatdiff_in_euler

from mujoco_panda.controllers.joint_torque_controllers import OSHybridForceMotionController # when controller is activated, step simulation is done automatically

if __name__ == "__main__":
    MODEL_PATH = os.environ['MJ_PANDA_PATH'] + \
        '/mujoco_panda/models/'
    p = PandaArm(render=True, model_path=MODEL_PATH+'panda_block_table.xml',
                 compensate_gravity=True, grav_comp_model_path=MODEL_PATH+'panda_block_table.xml')

    ctrl = OSHybridForceMotionController(p)

    render_rate = 100
    
    p.set_neutral_pose()
    p.step()
    time.sleep(0.01)

    new_pose = p._sim.data.qpos.copy()[:7]

    curr_ee, original_ori = p.ee_pose()

    target_z_traj = np.linspace(curr_ee[2], curr_ee[2]+0.1, 25).tolist()
    z_target = curr_ee[2]

    target_pos = curr_ee.copy()

    ctrl.set_active(True)

    now_r = time.time()
    i = 0
    while i < len(target_z_traj):
        z_target = target_z_traj[i]
        # print(p._sim.data.qfrc_applied[:] )
        robot_pos, robot_ori = p.ee_pose()
        elapsed_r = time.time() - now_r

        target_pos[2] = z_target
        ctrl.set_goal(target_pos, original_ori)

        if elapsed_r >= 0.1:
            i += 1
            now_r = time.time()
        render_frame(p.viewer, robot_pos, robot_ori)
        render_frame(p.viewer, target_pos, original_ori, alpha=0.2)

        p.render()
        
    print ("Done controlling")

    while True:
        robot_pos, robot_ori = p.ee_pose()
        render_frame(p.viewer, robot_pos, robot_ori)
        render_frame(p.viewer, target_pos, original_ori, alpha = 0.2)
        p.render()

    run_controller = False
    ctrl_thread.join()


