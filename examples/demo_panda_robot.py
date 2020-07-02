from mujoco_panda import PandaArm
from mujoco_panda.utils.debug_utils import ParallelPythonCmd

"""
Testing PandaArm instance and parallel command utility

# in the parallel debug interface run PandaArm arm commands to control and monitor the robot.
# eg: p.set_neutral_pose(), p.joint_angles(), p.ee_pose(), etc.

"""

def exec_func(cmd):
    if cmd == '':
        return None
    print(cmd)
    try:
        if "=" in cmd:
            exec(cmd)
            a = "Executed cmd: {}".format(cmd)
        else:
            a = eval(cmd)
            print(a)
    except Exception as e:
        a = "{}: {}".format(type(e).__name__, e)
    if a is not None:
        return str(a)


if __name__ == "__main__":
    
    p = PandaArm.fullRobotWithTorqueActuators(render=False,compensate_gravity=True)
    p.sim.step()
    p.start_asynchronous_run() # run simulation in separate independent thread

    # _cth = ParallelPythonCmd(exec_func)

    # while True:
    #     print (p.sim.data.time)
    #     p.render()

# in the parallel debug interface run PandaArm arm commands to control and monitor the robot.
# eg: p.set_neutral_pose(), p.joint_angles(), p.ee_pose(), etc.
