# File      :test.py
# Time      :2022/11/10--14:13
# Author    :JF Li
# Version   :python 3.7

import re
import numpy as np


def data_extraction(path, size):
    data = []
    with open(path, 'r', encoding = 'utf-8') as file:
        while True:
            line = file.readline()
            if not line:
                break
            if line[0] != '%':
                line = re.sub(' +', ' ', line)
                strline = line.split(' ', 4)
                numline = []
                for i in range(len(strline)):
                    numline.append(float(strline[i]))
                data.append(numline)
    data = np.array(data)

    # split interior and boundary points and reform them as structured data
    delta = 1 / (size - 1)
    coor_x = np.array([[i * delta for i in range(size)] for _ in range(size)])
    coor_y = np.array([[i * delta] * size for i in range(size)])
    coor = np.concatenate((np.expand_dims(coor_x, axis = 0), np.expand_dims(coor_y, axis = 0)), axis = 0)
    value = np.zeros((3, size, size), dtype = float)
    visit = np.zeros((size, size), dtype = float)
    valid = 0
    try:
        for i in range(data.shape[0]):
            index_x = int(data[i][0] * (size - 1) + 0.1)
            index_y = int(data[i][1] * (size - 1) + 0.1)
            if visit[index_y][index_x] != 0:
                raise Exception("set real value error!")
            else:
                value[0][index_y][index_x] = data[i][2]
                value[1][index_y][index_x] = data[i][3]
                value[2][index_y][index_x] = data[i][4]
                visit[index_y][index_x] = 1
                valid += 1
    except Exception as e:
        print(e)
    assert valid == (size * size), "size error!"
    return coor, value


def getDataset(coor, value, data_size):
    coor = coor.reshape(2, data_size ** 2).T
    value = value.reshape(3, data_size ** 2).T
    return coor, value


def frobenius_norm(array):
    return np.sqrt(np.sum(np.power(array, 2)))


def mae(array):
    return np.sum(np.abs(array)) / array.shape[0]


def test(value, value_pred):
    u = value[:, 0].reshape(value.shape[0], 1)
    v = value[:, 1].reshape(value.shape[0], 1)
    p = value[:, 2].reshape(value.shape[0], 1)
    u_pred = value_pred[:, 0].reshape(value.shape[0], 1)
    v_pred = value_pred[:, 1].reshape(value.shape[0], 1)
    p_pred = value_pred[:, 2].reshape(value.shape[0], 1)
    uv = (u ** 2 + v ** 2) ** 0.5
    uv_pred = (u_pred ** 2 + v_pred ** 2) ** 0.5
    p = p - np.mean(p)
    p_pred = p_pred - np.mean(p_pred)
    # error_uv = frobenius_norm(uv_pred - uv) / frobenius_norm(uv)
    # error_p = frobenius_norm(p_pred - p) / frobenius_norm(p)
    error_u = mae(u - u_pred)
    error_v = mae(v - v_pred)
    error_uv = mae(uv - uv_pred)
    error_p = mae(p - p_pred)
    return error_u, error_v, error_uv, error_p
