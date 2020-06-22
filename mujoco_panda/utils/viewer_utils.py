from mujoco_py.generated import const
from .tf import quat2Mat

def render_frame(viewer, pos, quat, scale = 0.1, alpha = 1.):
    """ 
    Visualise a 3D coordinate frame.
    """
    viewer.add_marker(pos=pos,
                      label='',
                      type=const.GEOM_SPHERE,
                      size=[.01, .01, .01])
    mat = quat2Mat(quat)
    cylinder_half_height = scale
    pos_cylinder = pos + mat.dot([0.0, 0.0, cylinder_half_height])
    viewer.add_marker(pos=pos_cylinder,
                      label='',
                      type=const.GEOM_CYLINDER,
                      size=[.005, .005, cylinder_half_height],
                      rgba=[0.,0.,1.,alpha],
                      mat=mat)
    pos_cylinder = pos + mat.dot([cylinder_half_height, 0.0, 0.])
    viewer.add_marker(pos=pos_cylinder,
                      label='',
                      type=const.GEOM_CYLINDER,
                      size=[cylinder_half_height, .005, .005 ],
                      rgba=[1., 0., 0., alpha],
                      mat=mat)
    pos_cylinder = pos + mat.dot([0.0, cylinder_half_height, 0.0])
    viewer.add_marker(pos=pos_cylinder,
                      label='',
                      type=const.GEOM_CYLINDER,
                      size=[.005, cylinder_half_height, .005],
                      rgba=[0., 1., 0., alpha],
                      mat=mat)


def render_point(viewer, pos):
    viewer.add_marker(pos=pos,
                      label='',
                      type=const.GEOM_SPHERE,
                      size=[.01, .01, .01])
