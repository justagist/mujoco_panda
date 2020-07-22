from mujoco_panda.controllers.controller_base import ControllerBase
from mujoco_panda.utils.tf import quatdiff_in_euler
from .configs import BASIC_HYB_CONFIG
import numpy as np

class OSHybridForceMotionController(ControllerBase):
    """
    Torque-based task-space hybrid force motion controller.
    Computes the joint torques required for achieving a desired
    end-effector pose and/or wrench. Goal values and directions
    are defined in cartesian coordinates.

    First computes cartesian force for achieving the goal using PD
    control law, then computes the corresponding joint torques using 
    :math:`\tau = J^T F`.
    
    """

    def __init__(self, robot_object, config=BASIC_HYB_CONFIG, control_rate=None, *args, **kwargs):
        """
        contstructor

        :param robot_object: the :py:class:`PandaArm` object to be controlled
        :type robot_object: PandaArm
        :param config: dictionary of controller parameters, defaults to 
            BASIC_HYB_CONFIG (see config for reference)
        :type config: dict, optional
        """
        if control_rate is not None:
            config['control_rate'] = control_rate
            
        super(OSHybridForceMotionController,
              self).__init__(robot_object, config)

        self._goal_pos, self._goal_ori = self._robot.ee_pose()
        self._goal_vel, self._goal_omg = np.zeros(3), np.zeros(3)
        self._goal_force, self._goal_torque = np.zeros(
            3), np.zeros(3)

        # gains for position
        self._kp_p = np.diag(self._config['kp_p'])
        self._kd_p = np.diag(self._config['kd_p'])

        # gains for orientation
        self._kp_o = np.diag(self._config['kp_o'])
        self._kd_o = np.diag(self._config['kd_o'])

        # gains for force
        self._kp_f = np.diag(self._config['kp_f'])
        self._kd_f = np.diag(self._config['kd_f'])

        # gains for torque
        self._kp_t = np.diag(self._config['kp_t'])
        self._kd_t = np.diag(self._config['kd_t'])

        self._force_dir = np.diag(self._config['ft_dir'][:3])
        self._torque_dir = np.diag(self._config['ft_dir'][3:])

        self._pos_p_dir = np.diag([1, 1, 1]) ^ self._force_dir
        self._pos_o_dir = np.diag([1, 1, 1]) ^ self._torque_dir

        self._use_null_ctrl = self._config['use_null_space_control']

        if self._use_null_ctrl:

            self._null_Kp = np.diag(self._config['null_kp'])

            self._null_ctrl_wt = self._config['null_ctrl_wt']

        self._pos_threshold = self._config['linear_error_thr']

        self._angular_threshold = self._config['angular_error_thr']

    def set_active(self, status=True):
        """
        Override parent method to reset goal values

        :param status: To deactivate controller, set False. Defaults to True.
        :type status: bool, optional
        """
        if status:
            self._robot.forward_sim()
            self._goal_pos, self._goal_ori = self._robot.ee_pose()
            self._goal_vel, self._goal_omg = np.zeros(3), np.zeros(3)
            self._goal_force, self._goal_torque = np.zeros(
                3), np.zeros(3)
        self._is_active = status

    def _compute_cmd(self):
        """
        Actual computation of command given the desired goal states

        :return: computed joint torque values
        :rtype: np.ndarray (7,)
        """
        # calculate the jacobian of the end effector
        jac_ee = self._robot.jacobian()

        # get position of the end-effector
        curr_pos, curr_ori = self._robot.ee_pose()
        curr_vel, curr_omg = self._robot.ee_velocity()

        # current end-effector force and torque measured
        curr_force, curr_torque = self._robot.get_ft_reading()

        # error values
        delta_pos = self._goal_pos - curr_pos
        delta_vel = self._goal_vel - curr_vel
        delta_force = self._force_dir.dot(self._goal_force - curr_force)
        delta_torque = self._torque_dir.dot(self._goal_torque - curr_torque)

        if self._goal_ori is not None:
            delta_ori = quatdiff_in_euler(curr_ori, self._goal_ori)
        else:
            delta_ori = np.zeros(delta_pos.shape)

        delta_ori = self._pos_o_dir.dot(delta_ori)

        delta_omg = self._pos_o_dir.dot(self._goal_omg - curr_omg)

        # compute force control part along force dimensions # negative sign to convert from experienced to applied
        force_control = -self._force_dir.dot(self._kp_f.dot(delta_force) - np.abs(self._kd_f.dot(delta_vel)) + self._goal_force)
        # compute position control force along position dimensions (orthogonal to force dims)
        position_control = self._pos_p_dir.dot(
            self._kp_p.dot(delta_pos) + self._kd_p.dot(delta_vel))

        # total cartesian force at end-effector
        x_des = position_control + force_control

        if self._goal_ori is not None:  # orientation conttrol
            
            if np.linalg.norm(delta_ori) < self._angular_threshold:
                delta_ori = np.zeros(delta_ori.shape)
                delta_omg = np.zeros(delta_omg.shape)

            # compute orientation control force along orientation dimensions
            ori_pos_ctrl = self._pos_o_dir.dot(
                self._kp_o.dot(delta_ori) + self._kd_o.dot(delta_omg))

            # compute torque control force along torque dimensions (orthogonal to orientation dimensions) 
            # negative sign to convert from experienced to applied
            torque_f_ctrl = - self._torque_dir.dot(self._kp_t.dot(
                delta_torque) - self._kd_t.dot(delta_omg) + self._goal_torque)

            # total torque in cartesian at end-effector
            omg_des = ori_pos_ctrl + torque_f_ctrl

        else:

            omg_des = np.zeros(3)

        f_ee = np.hstack([x_des, omg_des])  # Desired end-effector force

        u = np.dot(jac_ee.T, f_ee) # desired joint torque

        if np.any(np.isnan(u)):
            u = self._cmd
        else:
            self._cmd = u

        if self._use_null_ctrl: # null-space control, if required

            null_space_filter = self._null_Kp.dot(
                np.eye(7) - jac_ee.T.dot(np.linalg.pinv(jac_ee.T, rcond=1e-3)))

            # add null-space torque in the null-space projection of primary task
            self._cmd = self._cmd + \
                null_space_filter.dot(
                    self._robot._neutral_pose-self._robot.joint_positions()[:7])

        # update the error
        self._error = {'linear': delta_pos, 'angular': delta_ori}

        return self._cmd

    def set_goal(self, goal_pos, goal_ori=None, goal_vel=np.zeros(3), goal_omg=np.zeros(3), goal_force=None, goal_torque=None):
        """
        change the target for the controller
        """
        # self._mutex.acquire()
        self._goal_pos = goal_pos
        self._goal_ori = goal_ori
        self._goal_vel = goal_vel
        self._goal_omg = goal_omg
        if goal_force is not None:
            self._goal_force = - np.asarray(goal_force) # applied force = - experienced force
        if goal_torque is not None:
            self._goal_torque = - np.asarray(goal_torque) # applied torque = - experienced torque
        # self._mutex.release()

    def change_ft_dir(self, directions):
        """
        Change directions along which force/torque control is performed.

        :param directions: 6 binary values for [x,y,z,x_rot,y_rot,z_rot], 
            1 indicates force direction, 0 position. Eg: [0,0,1,0,0,1]
            means force control is along the cartesian Z axis and (torque)
            about the Z axis, while other dimensions are position (and 
            orientation) controlled.
        :type directions: [int] * 6
        """
        # self._mutex.acquire()
        self._force_dir = np.diag(directions[:3])

        self._torque_dir = np.diag(directions[3:])

        self._pos_p_dir = np.diag([1, 1, 1]) ^ self._force_dir

        self._pos_o_dir = np.diag([1, 1, 1]) ^ self._torque_dir
        # self._mutex.release()
