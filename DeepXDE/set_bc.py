# File      :set_bc.py
# Time      :2022/11/9--21:41
# Author    :JF Li
# Version   :python 3.7

import numpy as np

init_u = 1
inlet = [[0, 0.2], [0, 0.5]]
outlet = [[0.4, 1], [0.7, 1]]


# boundary conditions
def func_u_zero(coor):
    '''boundary condition of u'''
    return np.zeros((coor.shape[0], 1))


def func_u(coor):
    '''boundary condition of u'''
    return np.ones((coor.shape[0], 1)) * init_u


def func_v_zero(coor):
    '''boundary condition of v'''
    return np.zeros((coor.shape[0], 1))


def func_v(coor):
    '''Neumann boundary condition of v'''
    return np.zeros((coor.shape[0], 1))


def func_p(coor):
    '''boundary condition of p'''
    return np.zeros((coor.shape[0], 1))


# boundary positions
def pos_u_zero(coor, on_boundary):
    '''position of boundary of u_zero'''
    return on_boundary and not (np.isclose(coor[0], inlet[0][0]) and inlet[0][1] <= coor[1] <= inlet[1][1])


def pos_u(coor, on_boundary):
    '''position of boundary of u'''
    return on_boundary and np.isclose(coor[0], inlet[0][0]) and inlet[0][1] <= coor[1] <= inlet[1][1]


def pos_v_zero(coor, on_boundary):
    '''position of boundary of v_zero'''
    return on_boundary and not (np.isclose(coor[1], outlet[0][1]) and outlet[0][0] <= coor[0] <= outlet[1][0])


def pos_v(coor, on_boundary):
    '''position of boundary of v'''
    return on_boundary and np.isclose(coor[1], outlet[0][1]) and outlet[0][0] <= coor[0] <= outlet[1][0]


def pos_p(coor, on_boundary):
    '''position of boundary of p'''
    return on_boundary and np.isclose(coor[1], outlet[0][1]) and outlet[0][0] <= coor[0] <= outlet[1][0]
