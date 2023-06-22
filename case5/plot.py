# File      :plot.py
# Time      :2022/3/16--16:02
# Author    :JF Li
# Version   :python 3.7

import numpy as np
import torch
import matplotlib.pyplot as plt
from _2Dsteady_TopCapDrive_SUS import data_extraction, PICNN, test
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
    pred = test(coor, value, model, init_u, model_path)
    return coor, value, pred


def plottingAndCombine(coor, value, pred):
    # tensor to ndarray and post-processing
    coor = coor.detach().numpy()
    value4_u = value[0].detach().numpy()[0][0]
    value4_v = value[0].detach().numpy()[0][1]
    value7_u = value[1].detach().numpy()[0][0]
    value7_v = value[1].detach().numpy()[0][1]
    value8_u = value[2].detach().numpy()[0][0]
    value8_v = value[2].detach().numpy()[0][1]
    pred4_u = pred[0].detach().numpy()[0][0]
    pred4_v = pred[0].detach().numpy()[0][1]
    pred7_u = pred[1].detach().numpy()[0][0]
    pred7_v = pred[1].detach().numpy()[0][1]
    pred8_u = pred[2].detach().numpy()[0][0]
    pred8_v = pred[2].detach().numpy()[0][1]
    value4_uv = np.power(np.power(value4_u, 2) + np.power(value4_v, 2), 0.5)
    value7_uv = np.power(np.power(value7_u, 2) + np.power(value7_v, 2), 0.5)
    value8_uv = np.power(np.power(value8_u, 2) + np.power(value8_v, 2), 0.5)
    pred4_uv = np.power(np.power(pred4_u, 2) + np.power(pred4_v, 2), 0.5)
    pred7_uv = np.power(np.power(pred7_u, 2) + np.power(pred7_v, 2), 0.5)
    pred8_uv = np.power(np.power(pred8_u, 2) + np.power(pred8_v, 2), 0.5)
    x = coor[0][0]
    y = coor[0][1]
    uv4_min = min(value4_uv.flatten())
    uv4_max = max(value4_uv.flatten())
    uv7_min = min(value7_uv.flatten())
    uv7_max = max(value7_uv.flatten())
    uv8_min = min(value8_uv.flatten())
    uv8_max = max(value8_uv.flatten())

    # plotting
    fig, big_axes = plt.subplots(nrows = 2, ncols = 1, figsize = (9, 6), sharey = False, sharex = False)
    for i in range(2):
        big_axes[i].tick_params(labelcolor = (0., 0., 0., 0.), top = False, bottom = False, left = False, right = False)
        big_axes[i]._frameon = False

    big_axes[0].set_title('(a) Truth', y = -0.2, fontsize = 15)
    big_axes[1].set_title('(b) Ours', y = -0.2, fontsize = 15)

    # Pressure
    ax1 = fig.add_subplot(2, 3, 1)
    cf = ax1.scatter(x, y, c = value4_uv, alpha = 0.9, edgecolors = 'none',
                     cmap = 'coolwarm', marker = 's', s = 32, vmin = uv4_min, vmax = uv4_max)
    ax1.axis('square')
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    ax1.set_title('Re=100')
    fig.colorbar(cf, ax = ax1)

    ax2 = fig.add_subplot(2, 3, 2)
    cf = ax2.scatter(x, y, c = value7_uv, alpha = 0.9, edgecolors = 'none',
                     cmap = 'coolwarm', marker = 's', s = 32, vmin = uv7_min, vmax = uv7_max)
    ax2.axis('square')
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])
    ax2.set_title('Re=500')
    fig.colorbar(cf, ax = ax2)

    ax3 = fig.add_subplot(2, 3, 3)
    cf = ax3.scatter(x, y, c = value8_uv, alpha = 0.9, edgecolors = 'none',
                     cmap = 'coolwarm', marker = 's', s = 32, vmin = uv8_min, vmax = uv8_max)
    ax3.axis('square')
    ax3.set_xlim([0, 1])
    ax3.set_ylim([0, 1])
    ax3.set_title('Re=1000')
    fig.colorbar(cf, ax = ax3)

    ax4 = fig.add_subplot(2, 3, 4)
    cf = ax4.scatter(x, y, c = pred4_uv, alpha = 0.9, edgecolors = 'none',
                     cmap = 'coolwarm', marker = 's', s = 32, vmin = uv4_min, vmax = uv4_max)
    ax4.axis('square')
    ax4.set_xlim([0, 1])
    ax4.set_ylim([0, 1])
    ax4.set_title('Re=100')
    fig.colorbar(cf, ax = ax4)

    # Velocity
    ax5 = fig.add_subplot(2, 3, 5)
    cf = ax5.scatter(x, y, c = pred7_uv, alpha = 0.9, edgecolors = 'none',
                     cmap = 'coolwarm', marker = 's', s = 32, vmin = uv7_min, vmax = uv7_max)
    ax5.axis('square')
    ax5.set_xlim([0, 1])
    ax5.set_ylim([0, 1])
    ax5.set_title('Re=500')
    fig.colorbar(cf, ax = ax5)

    ax6 = fig.add_subplot(2, 3, 6)
    cf = ax6.scatter(x, y, c = pred8_uv, alpha = 0.9, edgecolors = 'none',
                     cmap = 'coolwarm', marker = 's', s = 32, vmin = uv8_min, vmax = uv8_max)
    ax6.axis('square')
    ax6.set_xlim([0, 1])
    ax6.set_ylim([0, 1])
    ax6.set_title('Re=1000')
    fig.colorbar(cf, ax = ax6)

    # save and show
    fig.set_facecolor('w')
    plt.tight_layout()
    plt.savefig('./figures/3.pdf', dpi = 600, format = 'pdf')
    plt.show()


def convergence(loss_path, save_path = './figures/loss.png'):
    data = pd.read_csv(loss_path, names = ['loss_u', 'loss_v', 'mass'])
    x = range(2000)
    plt.plot(x, data['mass'] + data['loss_u'] + data['loss_v'])
    plt.xlabel('Epoch')
    plt.ylabel('Error')
    plt.legend()
    plt.yscale('log')
    # plt.savefig(save_path)
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
    coor, value1, pred1 = loadAndPred('./data/data1.txt', 51, './models/data1_SUS/model_200000.pt')
    _, value2, pred2 = loadAndPred('./data/data2.txt', 51, './models/data2_SUS/model_200000.pt')
    _, value3, pred3 = loadAndPred('./data/data3.txt', 51, './models/data3_SUS/model_200000.pt')

    # # plotting
    pred = [pred1, pred2, pred3]
    value = [value1, value2, value3]
    plottingAndCombine(coor, value, pred)
    # lineGraph(value, pred)
    # convergence('./models/loss.txt')
