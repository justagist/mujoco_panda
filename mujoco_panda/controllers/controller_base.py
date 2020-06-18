import abc
import threading
import logging
import time
import numpy as np

LOG_LEVEL = "DEBUG"

class ControllerBase(object):

    def __init__(self, robot_object, config={}, run_step_always=True):

        self._robot = robot_object

        logging.basicConfig(format='\n{}: %(levelname)s: %(message)s\n'.format(
            self.__class__.__name__), level=LOG_LEVEL)
        self._logger = logging.getLogger(__name__)

        self._config = config
        
        if 'control_rate' in self._config:
            control_rate = self._config['control_rate']
        else:
            control_rate = 1./self._robot.model.opt.timestep


        self._is_active = False

        self._cmd = self._robot.sim.data.ctrl[self._robot.actuated_arm_joints].copy(
        )

        self._mutex = threading.Lock()

        self._ctrl_thread = threading.Thread(target = self._send_cmd, args=[control_rate])
        self._is_running = True

        self._ctrl_thread.start()

        self._error = {'linear': np.zeros(3), 'angular': np.zeros(3)}

    @property
    def is_active(self):
        return self._is_active

    def set_active(self, status = True):
        self._is_active = status

    def toggle_activate(self):
        self.set_active(status = not self._is_active)

    @abc.abstractmethod
    def _compute_cmd(self):
        raise NotImplementedError("Method must be implemented in child class!")

    @abc.abstractmethod
    def set_goal(self, *args, **kwargs):
        raise NotImplementedError("Method must be implemented in child class!")

    def _send_cmd(self, control_rate):
        while self._is_running:
            if self._is_active:
                now_c = time.time()
                self._mutex.acquire()
                self._compute_cmd()
                self._robot.set_joint_commands(
                    self._cmd, joints=self._robot.actuated_arm_joints, compensate_dynamics=True)
                self._robot.step(render=False)
                self._mutex.release()

                elapsed_c = time.time() - now_c
                sleep_time_c = (1./control_rate) - elapsed_c
                if sleep_time_c > 0.0:
                    time.sleep(sleep_time_c)

    def __del__(self):
        self._is_running = False
        self._ctrl_thread.join()
