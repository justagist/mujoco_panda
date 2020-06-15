#!/usr/bin/env python3

from mujoco_py import load_model_from_path, MjSim, MjViewer
from mujoco_py.modder import TextureModder
import os

model_filename = os.environ['MJ_PANDA_PATH']+'/mujoco_panda/models/franka_panda.xml'
model = load_model_from_path(model_filename)
sim = MjSim(model)

print (sim.get_state())
viewer = MjViewer(sim)

t = 0

while True:
    # for name in sim.model.geom_names:
    #     modder.rand_all(name)
    # sim.data.ctrl[0] += 0.001
    # sim.data.ctrl[1] += 0.001
    # sim.data.ctrl[2] += 0.001
    # sim.data.ctrl[3] += 0.001
    # sim.data.ctrl[4] += 0.001
    # sim.data.ctrl[5] += 0.001
    # sim.data.ctrl[6] += 0.001
    # sim.data.ctrl[8] += 0.1
    # sim.data.ctrl[7] += 0.1
    # print (len(sim.data.qpos))

    # sim.data.qfrc_applied[8] += 1.
    # sim.data.qfrc_applied[7] += 1.
    # print (len(sim.data.qpos))
    sim.step()
    viewer.render()
    # sim.data.qfrc_applied[0] = 10.
    t += 1
    if t > 100:
        break
