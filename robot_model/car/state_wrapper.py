import torch
from typing import Dict
from collections import namedtuple

# STATE_DEF_LIST = ['v_x', 'v_y', 'r', 'omega_wheels', 'friction',
#                   'delta', 'Iq', 'ax_imu', 'ay_imu', 'r_imu']
# STATE_DEF_LIST_SHORT = STATE_DEF_LIST[:7]


WX = namedtuple('WX', ['v_x', 'v_y', 'r', 'friction',
                       'omega_wheels', 'delta'])

# @torch.compile
def StateWrapper(xu):
    ## NOT THE SAME AS IN UKF CODE
    return WX(v_x=xu[..., 0],
              v_y=xu[..., 1],
              r=xu[..., 2],
              friction=xu[..., 3],
              omega_wheels=xu[..., 4],
              delta=xu[..., 5])
