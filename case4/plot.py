# File      :plot.py
# Time      :2022/3/28--17:27
# Author    :JF Li
# Version   :python 3.7

import numpy as np
import torch
import matplotlib.pyplot as plt
from _2Dsteady_ForcedConvectionHeatTransfer import data_extraction, PICNN, test
import pandas as pd


def loadAndPred(data_path, data_size, model_path):
    # load data
    coor, value = data_extraction(data_path, data_size)
    coor = torch.tensor(coor, dtype = torch.float32).resize_((1, coor.shape[0], coor.shape[1], coor.shape[2]))
    value = torch.tensor(value, dtype = torch.float32).resize_((1, value.shape[0], value.shape[1], value.shape[2]))
    # set model and train
    model = PICNN(upsample_size = [data_size - 2, data_size - 2])
    # test
    delta_xy = 0.02
    inlet = [[0, 0.2], [0, 0.7]]
    dirichlet_boundary = [20, 60]  # inlet, up/down/right
    pred = test(coor, value, model, delta_xy, inlet, dirichlet_boundary, model_path)
    return coor, value, pred


def plottingAndCombine(coor, value, pred):
    # tensor to ndarray and post-processing
    coor = coor.detach().numpy()
    value = value.detach().numpy()[0][0]
    predCD = pred[0].detach().numpy()[0][0] + 273.15
    predSUS = pred[1].detach().numpy()[0][0] + 273.15
    x = coor[0][0]
    y = coor[0][1]
    t_min = min(value.flatten())
    t_max = max(value.flatten())

    # plotting
    fig, big_axes = plt.subplots(nrows = 1, ncols = 1, figsize = (9, 3), sharey = False, sharex = False)
    big_axes.tick_params(labelcolor = (0., 0., 0., 0.), top = False, bottom = False, left = False, right = False)
    big_axes._frameon = False

    ax1 = fig.add_subplot(1, 3, 1)
    cf = ax1.scatter(x, y, c = value, alpha = 0.9, edgecolors = 'none',
                     cmap = 'rainbow', marker = 's', s = 32, vmin = t_min, vmax = t_max)
    ax1.axis('square')
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    ax1.set_title('Truth')
    fig.colorbar(cf, ax = ax1)

    ax2 = fig.add_subplot(1, 3, 2)
    cf = ax2.scatter(x, y, c = predCD, alpha = 0.9, edgecolors = 'none',
                     cmap = 'rainbow', marker = 's', s = 32, vmin = t_min, vmax = t_max)
    ax2.axis('square')
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])
    ax2.set_title('CD')
    fig.colorbar(cf, ax = ax2)

    ax3 = fig.add_subplot(1, 3, 3)
    cf = ax3.scatter(x, y, c = predSUS, alpha = 0.9, edgecolors = 'none',
                     cmap = 'rainbow', marker = 's', s = 32, vmin = t_min, vmax = t_max)
    ax3.axis('square')
    ax3.set_xlim([0, 1])
    ax3.set_ylim([0, 1])
    ax3.set_title('SUS')
    fig.colorbar(cf, ax = ax3)

    # save and show
    fig.set_facecolor('w')
    plt.tight_layout()
    plt.savefig('./figures/result_1.pdf', dpi = 600, format = 'pdf')
    # plt.show()


def convergence(loss_path, save_path):
    data = pd.read_csv(loss_path, names = ['loss_T'])
    x = range(2000)
    plt.plot(x, data['loss_T'])
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.yscale('log')
    plt.savefig(save_path)
    plt.show()


def lineGraph(value, pred):
    predCD = pred[0].detach().numpy()[0][0] + 273.15
    predSUS = pred[1].detach().numpy()[0][0] + 273.15
    real = value[0, 0, 32, :]
    cd = predCD[32, :]
    sus = predSUS[32, :]
    x = np.array([i * 0.02 for i in range(51)])

    # plotting
    fig, big_axes = plt.subplots(nrows = 1, ncols = 1, figsize = (8, 4), sharey = False, sharex = False)
    big_axes.tick_params(labelcolor = (0., 0., 0., 0.), top = False, bottom = False, left = False, right = False)
    big_axes._frameon = False

    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(x, real, label = 'Truth')
    ax1.plot(x, cd, label = 'CD')
    ax1.set_xlabel('x', fontdict = {'size': 12})
    ax1.set_ylabel('Temperature', fontdict = {'size': 12})
    ax1.legend()

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(x, real, label = 'Truth')
    ax2.plot(x, sus, label = 'SUS')
    ax2.set_xlabel('x', fontdict = {'size': 12})
    ax2.set_ylabel('Temperature', fontdict = {'size': 12})
    ax2.legend()

    # save and show
    fig.set_facecolor('w')
    plt.tight_layout()
    plt.savefig('./figures/result1_1.pdf', dpi = 600, format = 'pdf')
    # plt.show()


if __name__ == '__main__':
    # load predictions
    coor, value, predCD = loadAndPred('./data/data7.txt', 51, './models/data7_CD/model_200000.pt')
    _, _, predSUS = loadAndPred('./data/data7.txt', 51, './models/data7_SUS/model_200000.pt')

    # plotting
    pred = [predCD, predSUS]
    plottingAndCombine(coor, value, pred)
    lineGraph(value, pred)
