import os
import time
import mujoco_py
import numpy as np
from mujoco_panda import PandaArm
from mujoco_panda.utils.viewer_utils import render_frame
from mujoco_panda.utils.debug_utils import ParallelPythonCmd
from mujoco_panda.controllers.torque_based_controllers import OSHybridForceMotionController


def exec_func(cmd):
    if cmd == '':
        return None
    a = eval(cmd)
    print(cmd)
    print(a)
    if a is not None:
        return str(a)

MODEL_PATH = os.environ['MJ_PANDA_PATH'] + \
    '/mujoco_panda/models/'


# controller parameters
KP_P = np.array([7000., 7000., 7000.])
KP_O = np.array([3000., 3000., 3000.])
ctrl_config = {
    'kp_p': KP_P,
    'kd_p': 2.*np.sqrt(KP_P),
    'kp_o': KP_O,  # 10gains for orientation
    'kd_o': [1., 1., 1.],  # gains for orientation
    'kp_f': [1., 1., 1.],  # gains for force
    'kd_f': [0., 0., 0.],  # 25gains for force
    'kp_t': [1., 1., 1.],  # gains for torque
    'kd_t': [0., 0., 0.],  # gains for torque
    'alpha': 3.14*0,
    'use_null_space_control': True,
    'ft_dir': [0, 0, 0, 0, 0, 0],
    # newton meter
    'null_kp': [5.0, 10.0, 5.0, 10.0, 5.0, 10.0, 5.0],
    'null_kd': 0,
    'null_ctrl_wt': 2.5,
    'use_orientation_ctrl': True,
    'linear_error_thr': 0.025,
    'angular_error_thr': 0.01,
}

if __name__ == "__main__":
    p = PandaArm(model_path=MODEL_PATH+'panda_block_table.xml',
                 render=True, compensate_gravity=False, smooth_ft_sensor=True)

    if mujoco_py.functions.mj_isPyramidal(p.model):
        print("Type of friction cone is pyramidal")
    else:
        print("Type of friction cone is eliptical")

    # cmd = ParallelPythonCmd(exec_func)
    p.set_neutral_pose()
    p.step()
    time.sleep(1.0)
    
    # create controller instance with default controller gains
    ctrl = OSHybridForceMotionController(p, config=ctrl_config)    

    # --- define trajectory in position -----
    curr_ee, curr_ori = p.ee_pose()
    goal_pos_1 = curr_ee.copy()
    goal_pos_1[2] = 0.58
    goal_pos_1[0] += 0.15
    goal_pos_1[1] -= 0.2
    target_traj_1 = np.linspace(curr_ee, goal_pos_1, 100)
    z_target = curr_ee[2]
    target_ori = np.asarray([0, -0.924, -0.383, 0], dtype=np.float64)
    goal_pos_2 = target_traj_1[-1, :].copy()
    goal_pos_2[2] = 0.55
    wait_traj = np.asarray([target_traj_1[-1, :], ]*30)
    target_traj_2 = np.linspace(target_traj_1[-1, :], np.asarray(
        [goal_pos_2[0], goal_pos_2[1], goal_pos_2[2]]), 20)
    goal_pos_3 = target_traj_2[-1, :].copy()
    goal_pos_3[1] += 0.4
    target_traj_3 = np.linspace(target_traj_2[-1, :], goal_pos_3, 200)
    target_traj = np.vstack(
        [target_traj_1, wait_traj, target_traj_2, target_traj_3])
    # --------------------------------------

    ctrl.set_active(True) # activate controller (simulation step and controller thread now running)

    now_r = time.time()
    i = 0
    count = 0
    while i < target_traj.shape[0]:
        # get current robot end-effector pose
        robot_pos, robot_ori = p.ee_pose() 

        elapsed_r = time.time() - now_r

        # render controller target and current ee pose using frames
        render_frame(p.viewer, robot_pos, robot_ori)
        render_frame(p.viewer, target_traj[i, :], target_ori, alpha=0.2)

        if i == 130: # activate force control when the robot is near the table
            ctrl.change_ft_dir([0,0,1,0,0,0]) # start force control along Z axis
            ctrl.set_goal(target_traj[i, :],
                          target_ori, goal_force=[0, 0, -25]) # target force in cartesian frame
        else:
            ctrl.set_goal(target_traj[i, :], target_ori)

        if elapsed_r >= 0.1:
            i += 1 # change target less frequently compared to render rate
            print ("smoothed FT reading: ", p.get_ft_reading(pr=True))
            now_r = time.time()

        p.render() # render the visualisation
    
    input("Trajectory complete. Hit Enter to deactivate controller")
    ctrl.set_active(False)
    ctrl.stop_controller_cleanly()
