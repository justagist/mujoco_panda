import numpy as np

KP_P = np.array([15000., 15000., 15000.])
KP_O = np.array([7000., 7000., 7000.])
BASIC_HYB_CONFIG = {
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
