#!/usr/bin/env python

from setuptools import setup
# This setup is suitable for "python setup.py develop".
import os

matches = []
for root, dirnames, filenames in os.walk("mujoco_panda/models"):
      for filename in filenames:
            matches.append(os.path.join(root.replace("mujoco_panda/",""), filename))

setup(name='mujoco_panda',
      version='0.1',
      description='Model definitions and Python API for controlling a Franka Emika Panda robot in the mujoco physics simulator.',
      author='Saif Sidhik',
      maintainer="Florent Audonnet",
      maintainer_email="f.audonnet.1@research.gla.ac.uk",
      author_email='sxs1412@bham.ac.uk',
      url='https://github.com/09ubberboy90/mujoco_panda',
      packages=['mujoco_panda', "mujoco_panda.utils", "mujoco_panda.controllers.torque_based_controllers", "mujoco_panda.controllers"],
      install_requires=['numpy', 'scipy', 'numpy-quaternion', 'mujoco-py @ git+https://github.com/openai/mujoco-py'],
      package_data={'mujoco_panda': matches}
      )
