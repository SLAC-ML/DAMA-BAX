import numpy as np
from scipy.io import loadmat
from .dama_resources import get_ma_config, get_momentum_grid


def gen_ray_angles(a=5):
    
    x1 = np.arange(10)
    theta1 = (10 - a) * x1 + a / 9 * x1 ** 2
    
    x2 = np.arange(1, 10)
    theta2 = 90 + (10 + a) * x2 - a / 9 * x2 ** 2
    
    theta = np.concatenate((theta1, theta2)) * np.pi / 180
    
    return theta


def gen_ray_grid_xy(npts_per_ray=72, a=5, x0_min=0.004, x0_max=0.025, y0_max=0.012):
    radl = x0_min + np.arange(npts_per_ray) / npts_per_ray * (x0_max - x0_min)
    theta = gen_ray_angles(a=a)
    
    rr, aa = np.meshgrid(radl, theta)
    xx = rr * np.cos(aa)
    yy = rr * np.sin(aa) * y0_max / x0_max
    
    return xx, yy


def gen_ray_grid_ar(npts_per_ray=72, a=5, x0_min=0.004, x0_max=0.025, y0_max=0.012):
    xx, yy = gen_ray_grid_xy(npts_per_ray, a, x0_min, x0_max, y0_max)
    aa = np.arctan2(yy, xx)
    rr = np.sqrt(xx ** 2 + yy ** 2)
    
    return aa, rr


def gen_spos():
    s = loadmat(str(get_ma_config()))['spos'].flatten()

    return s


def gen_momentum():
    # m = np.hstack([np.arange(-0.075, -0.029, 0.002), np.arange(0.031, 0.077, 0.002)])
    # m = np.hstack([np.arange(-0.075, -0.027, 0.002), [0], np.arange(0.029, 0.077, 0.002)])
    m = loadmat(str(get_momentum_grid()))['dpp0'][0]

    return m


def gen_grid_sm():
    s = np.arange(1, len(gen_spos()) + 1)
    m = gen_momentum()
    
    mm, ss = np.meshgrid(m, s)
    
    return ss, mm


def find_row_indices(selected, base):
    # For each row in selected, find its index in base
    
    indices = np.where((base==selected[:, None]).all(-1))[1]
    
    return indices


def find_element_indices(selected, base):
    
    differences = np.abs(selected[:, None] - base)
    indices = np.argmin(differences, axis=1)
    
    return indices
