import mujoco_py
from mujoco_panda import PandaArm
import numpy as np
from numpy import linalg as LA
import time
import threading
import quaternion
from mujoco_panda.utils.viewer_utils import render_frame

# --------- Modify as required ------------
# Task-space controller parameters
# stiffness gains
P_pos = 600.
P_ori = 50.
# damping gains
D_pos = 2.*np.sqrt(P_pos)
D_ori = 1.
# -----------------------------------------

def quatdiff_in_euler(quat_curr, quat_des):
    """
        Compute difference between quaternions and return 
        Euler angles as difference
    """
    # print (type(quat_curr), type(quat_des))
    curr_mat = quaternion.as_rotation_matrix(quaternion.from_float_array(quat_curr))
    des_mat = quaternion.as_rotation_matrix(quaternion.from_float_array(quat_des))
    rel_mat = des_mat.T.dot(curr_mat)
    rel_quat = quaternion.from_rotation_matrix(rel_mat)
    vec = quaternion.as_float_array(rel_quat)[1:]
    if rel_quat.w < 0.0:
        vec = -vec

    return -des_mat.dot(vec)


def compute_ts_force(curr_pos, curr_ori, goal_pos, goal_ori, curr_vel, curr_omg):
    delta_pos = (goal_pos - curr_pos).reshape([3, 1])
    delta_ori = quatdiff_in_euler(curr_ori, goal_ori).reshape([3, 1])
    # print 
    F = np.vstack([P_pos*(delta_pos), P_ori*(delta_ori)]) - \
        np.vstack([D_pos*(curr_vel).reshape([3,1]), D_ori*(curr_omg).reshape([3,1])])

    error = np.linalg.norm(delta_pos) + np.linalg.norm(delta_ori)

    return F, error

def controller_thread(ctrl_rate):
    global p, target_pos

    threshold = 0.005
    
    target_pos = curr_ee.copy()
    while run_controller:
        error = 100.
        while error > threshold:
            now_c = time.time()
            curr_pos, curr_ori = p.ee_pose()
            curr_vel, curr_omg = p.ee_velocity()

            target_pos[2] = z_target
            F, error = compute_ts_force(
                curr_pos, curr_ori, target_pos, original_ori, curr_vel, curr_omg)

            if error <= threshold:
                break

            impedance_acc_des = np.dot(p.jacobian().T, F).flatten().tolist()

            p.set_joint_commands(impedance_acc_des)
            # print (p.sim.data.qfrc_applied)
            p.step(render=False)

            elapsed_c = time.time() - now_c
            sleep_time_c = (1./ctrl_rate) - elapsed_c
            if sleep_time_c > 0.0:
                time.sleep(sleep_time_c)
        # print ("Yes")


if __name__ == "__main__":
    p = PandaArm(render = True)

    ctrl_rate = 1/p.model.opt.timestep

    render_rate = 100
    
    p.step(render=False)
    p.set_neutral_pose()
    p.step(render=False)

    new_pose = p._sim.data.qpos.copy()[:7]

    curr_ee, original_ori = p.ee_pose()

    target_z_traj = np.linspace(curr_ee[2], curr_ee[2]+0.1, 25).tolist()
    z_target = curr_ee[2]

    run_controller = True
    ctrl_thread = threading.Thread(target=controller_thread, args=[ctrl_rate])
    ctrl_thread.start()


    now_r = time.time()
    i = 0
    while i < len(target_z_traj):
        z_target = target_z_traj[i]
        # print(p._sim.data.qfrc_applied[:] )

        elapsed_r = time.time() - now_r
        # sleep_time_r = (1./render_rate) - elapsed_r
        # if sleep_time_r > 0.0:
        if elapsed_r >= 0.1:
            i += 1
            now_r = time.time()
        render_frame(p.viewer, target_pos, original_ori)
        p.render()

        #     time.sleep(sleep_time_r)
        
    print ("Done controlling")
    while True:
        render_frame(p.viewer, target_pos, original_ori)
        p.render()

    run_controller = False
    ctrl_thread.join()


