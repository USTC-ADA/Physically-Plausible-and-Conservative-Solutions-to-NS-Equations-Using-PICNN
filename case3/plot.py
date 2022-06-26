# File      :plot.py
# Time      :2022/3/16--16:02
# Author    :JF Li
# Version   :python 3.7

import numpy as np
import torch
import matplotlib.pyplot as plt
from _2Dsteady_NS import data_extraction, PICNN, test
import pandas as pd


def loadAndPred(data_path, data_size, model_path):
    # load data
    coor, value = data_extraction(data_path, data_size)
    coor = torch.tensor(coor, dtype = torch.float32).resize_((1, coor.shape[0], coor.shape[1], coor.shape[2]))
    value = torch.tensor(value, dtype = torch.float32).resize_((1, value.shape[0], value.shape[1], value.shape[2]))
    # set model and train
    model = PICNN(upsample_size = [data_size - 2, data_size - 2])
    # test
    init_u = 1
    delta_xy = 0.02
    inlet = [[0, 0.2], [0, 0.5]]
    outlet = [[0.4, 1], [0.7, 1]]
    pred = test(coor, value, model, delta_xy, init_u, inlet, outlet, model_path)
    return coor, value, pred


def plottingAndCombine(coor, value, pred):
    # tensor to ndarray and post-processing
    coor = coor.detach().numpy()
    value_u = value.detach().numpy()[0][0]
    value_v = value.detach().numpy()[0][1]
    value_p = value.detach().numpy()[0][2]
    predCD_u = pred[0].detach().numpy()[0][0]
    predCD_v = pred[0].detach().numpy()[0][1]
    predCD_p = pred[0].detach().numpy()[0][2]
    predFUS_u = pred[1].detach().numpy()[0][0]
    predFUS_v = pred[1].detach().numpy()[0][1]
    predFUS_p = pred[1].detach().numpy()[0][2]
    predSUS_u = pred[2].detach().numpy()[0][0]
    predSUS_v = pred[2].detach().numpy()[0][1]
    predSUS_p = pred[2].detach().numpy()[0][2]
    value_uv = np.power(np.power(value_u, 2) + np.power(value_v, 2), 0.5)
    predCD_uv = np.power(np.power(predCD_u, 2) + np.power(predCD_v, 2), 0.5)
    predFUS_uv = np.power(np.power(predFUS_u, 2) + np.power(predFUS_v, 2), 0.5)
    predSUS_uv = np.power(np.power(predSUS_u, 2) + np.power(predSUS_v, 2), 0.5)
    x = coor[0][0]
    y = coor[0][1]
    uv_min = min(value_uv.flatten())
    uv_max = max(value_uv.flatten())
    p_min = 0
    p_max = 0.6
    predCD_p_min = -0.06
    predCD_p_max = 0.54
    predFUS_p_min = 0.26
    predFUS_p_max = 0.86
    predSUS_p_min = 0.23
    predSUS_p_max = 0.83

    # plotting
    fig, big_axes = plt.subplots(nrows = 2, ncols = 1, figsize = (12, 6), sharey = False, sharex = False)
    for i in range(2):
        big_axes[i].tick_params(labelcolor = (0., 0., 0., 0.), top = False, bottom = False, left = False, right = False)
        big_axes[i]._frameon = False

    big_axes[0].set_title('(a) Pressure', y = -0.2, fontsize = 15)
    big_axes[1].set_title('(b) Velocity', y = -0.2, fontsize = 15)

    # Pressure
    ax1 = fig.add_subplot(2, 4, 1)
    cf = ax1.scatter(x, y, c = value_p, alpha = 0.9, edgecolors = 'none',
                     cmap = 'coolwarm', marker = 's', s = 32, vmin = p_min, vmax = p_max)
    ax1.axis('square')
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    ax1.set_title('Truth')
    fig.colorbar(cf, ax = ax1)

    ax2 = fig.add_subplot(2, 4, 2)
    cf = ax2.scatter(x, y, c = predCD_p, alpha = 0.9, edgecolors = 'none',
                     cmap = 'coolwarm', marker = 's', s = 32, vmin = predCD_p_min, vmax = predCD_p_max)
    ax2.axis('square')
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])
    ax2.set_title('CD')
    fig.colorbar(cf, ax = ax2)

    ax3 = fig.add_subplot(2, 4, 3)
    cf = ax3.scatter(x, y, c = predFUS_p, alpha = 0.9, edgecolors = 'none',
                     cmap = 'coolwarm', marker = 's', s = 32, vmin = predFUS_p_min, vmax = predFUS_p_max)
    ax3.axis('square')
    ax3.set_xlim([0, 1])
    ax3.set_ylim([0, 1])
    ax3.set_title('FUS')
    fig.colorbar(cf, ax = ax3)

    ax4 = fig.add_subplot(2, 4, 4)
    cf = ax4.scatter(x, y, c = predSUS_p, alpha = 0.9, edgecolors = 'none',
                     cmap = 'coolwarm', marker = 's', s = 32, vmin = predSUS_p_min, vmax = predSUS_p_max)
    ax4.axis('square')
    ax4.set_xlim([0, 1])
    ax4.set_ylim([0, 1])
    ax4.set_title('SUS')
    fig.colorbar(cf, ax = ax4)

    # Velocity
    ax5 = fig.add_subplot(2, 4, 5)
    cf = ax5.scatter(x, y, c = value_uv, alpha = 0.9, edgecolors = 'none',
                     cmap = 'coolwarm', marker = 's', s = 32, vmin = uv_min, vmax = uv_max)
    ax5.axis('square')
    ax5.set_xlim([0, 1])
    ax5.set_ylim([0, 1])
    ax5.set_title('Truth')
    fig.colorbar(cf, ax = ax5)

    ax6 = fig.add_subplot(2, 4, 6)
    cf = ax6.scatter(x, y, c = predCD_uv, alpha = 0.9, edgecolors = 'none',
                     cmap = 'coolwarm', marker = 's', s = 32, vmin = uv_min, vmax = uv_max)
    ax6.axis('square')
    ax6.set_xlim([0, 1])
    ax6.set_ylim([0, 1])
    ax6.set_title('CD')
    fig.colorbar(cf, ax = ax6)

    ax7 = fig.add_subplot(2, 4, 7)
    cf = ax7.scatter(x, y, c = predFUS_uv, alpha = 0.9, edgecolors = 'none',
                     cmap = 'coolwarm', marker = 's', s = 32, vmin = uv_min, vmax = uv_max)
    ax7.axis('square')
    ax7.set_xlim([0, 1])
    ax7.set_ylim([0, 1])
    ax7.set_title('FUS')
    fig.colorbar(cf, ax = ax7)

    ax8 = fig.add_subplot(2, 4, 8)
    cf = ax8.scatter(x, y, c = predSUS_uv, alpha = 0.9, edgecolors = 'none',
                     cmap = 'coolwarm', marker = 's', s = 32, vmin = uv_min, vmax = uv_max)
    ax8.axis('square')
    ax8.set_xlim([0, 1])
    ax8.set_ylim([0, 1])
    ax8.set_title('SUS')
    fig.colorbar(cf, ax = ax8)

    # save and show
    fig.set_facecolor('w')
    plt.tight_layout()
    plt.savefig('./figures/result_1.pdf', dpi = 600, format = 'pdf')
    # plt.show()


def convergence(loss_path, save_path):
    data = pd.read_csv(loss_path, names = ['loss_u', 'loss_v', 'mass'])
    x = range(2000)
    plt.plot(x, data['mass'] + data['loss_u'] + data['loss_v'])
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.yscale('log')
    plt.savefig(save_path)
    plt.show()


def lineGraph(value, pred):
    value_uv = torch.pow((torch.pow(value[0][0], 2) + torch.pow(value[0][1], 2)), 0.5)
    predCD_uv = torch.pow((torch.pow(pred[0][0][0], 2) + torch.pow(pred[0][0][1], 2)), 0.5)
    predFUS_uv = torch.pow((torch.pow(pred[1][0][0], 2) + torch.pow(pred[1][0][1], 2)), 0.5)
    predSUS_uv = torch.pow((torch.pow(pred[2][0][0], 2) + torch.pow(pred[2][0][1], 2)), 0.5)

    real_up = value_uv[-3, :].detach().numpy()
    predCD_up = predCD_uv[-3, :].detach().numpy()
    predFUS_up = predFUS_uv[-3, :].detach().numpy()
    predSUS_up = predSUS_uv[-3, :].detach().numpy()
    real_mid = value_uv[:, 25].detach().numpy()
    predCD_mid = predCD_uv[:, 25].detach().numpy()
    predFUS_mid = predFUS_uv[:, 25].detach().numpy()
    predSUS_mid = predSUS_uv[:, 25].detach().numpy()
    x = np.array([i * 0.02 for i in range(51)])

    # plotting
    fig, big_axes = plt.subplots(nrows = 2, ncols = 1, figsize = (10, 8), sharey = False, sharex = False)
    for i in range(2):
        big_axes[i].tick_params(labelcolor = (0., 0., 0., 0.), top = False, bottom = False, left = False, right = False)
        big_axes[i]._frameon = False

    big_axes[0].set_title('(a) y=0.96', y = -0.4, fontsize = 20)
    big_axes[1].set_title('(b) x=0.50', y = -0.4, fontsize = 20)

    # y=0.96
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.plot(x, real_up, label = 'Truth')
    ax1.plot(x, predCD_up, label = 'CD')
    ax1.set_xlabel('x', fontdict = {'size': 15})
    ax1.set_ylabel('Velocity', fontdict = {'size': 15})
    ax1.legend()

    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(x, real_up, label = 'Truth')
    ax2.plot(x, predFUS_up, label = 'FUS')
    ax2.set_xlabel('x', fontdict = {'size': 15})
    ax2.set_ylabel('Velocity', fontdict = {'size': 15})
    ax2.legend()

    ax3 = fig.add_subplot(2, 3, 3)
    ax3.plot(x, real_up, label = 'Truth')
    ax3.plot(x, predSUS_up, label = 'SUS')
    ax3.set_xlabel('x', fontdict = {'size': 15})
    ax3.set_ylabel('Velocity', fontdict = {'size': 15})
    ax3.legend()

    # x=0.50
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.plot(x, real_mid, label = 'Truth')
    ax4.plot(x, predCD_mid, label = 'CD')
    ax4.set_xlabel('y', fontdict = {'size': 15})
    ax4.set_ylabel('Velocity', fontdict = {'size': 15})
    ax4.legend()

    ax5 = fig.add_subplot(2, 3, 5)
    ax5.plot(x, real_mid, label = 'Truth')
    ax5.plot(x, predFUS_mid, label = 'FUS')
    ax5.set_xlabel('y', fontdict = {'size': 15})
    ax5.set_ylabel('Velocity', fontdict = {'size': 15})
    ax5.legend()

    ax6 = fig.add_subplot(2, 3, 6)
    ax6.plot(x, real_mid, label = 'Truth')
    ax6.plot(x, predSUS_mid, label = 'SUS')
    ax6.set_xlabel('y', fontdict = {'size': 15})
    ax6.set_ylabel('Velocity', fontdict = {'size': 15})
    ax6.legend()

    # save and show
    fig.set_facecolor('w')
    plt.tight_layout()
    plt.savefig('./figures/result1_1.pdf', dpi = 600, format = 'pdf')
    # plt.show()


if __name__ == '__main__':
    # load predictions
    coor, value, predCD = loadAndPred('./data/data8.txt', 51, './models/data8_CD/model_200000.pt')
    _, _, predFUS = loadAndPred('./data/data8.txt', 51, './models/data8/model_200000.pt')
    _, _, predSUS = loadAndPred('./data/data8.txt', 51, './models/data8_SUS/model_200000.pt')

    # plotting
    pred = [predCD, predFUS, predSUS]
    plottingAndCombine(coor, value, pred)
    lineGraph(value, pred)
