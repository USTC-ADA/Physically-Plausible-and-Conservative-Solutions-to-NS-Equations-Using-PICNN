# File      :plot.py
# Time      :2022/5/26--20:01
# Author    :JF Li
# Version   :python 3.7

import torch
import matplotlib.pyplot as plt
from _2Dsteady_convection_diffusion import data_extraction, PICNN, test


def loadAndPred(data_path, data_size, model_path):
    # load data
    coor, value = data_extraction(data_path, data_size)
    coor = torch.tensor(coor, dtype = torch.float32).resize_((1, coor.shape[0], coor.shape[1], coor.shape[2]))
    value = torch.tensor(value, dtype = torch.float32).resize_((1, value.shape[0], value.shape[1], value.shape[2]))
    # set model and train
    model = PICNN(upsample_size = data_size - 2)
    # test
    dirichlet_boundary = [10, 7, 5, 1]  # up down left right
    pred = test(coor, value, model, dirichlet_boundary, model_path)
    return coor, value, pred


def plottingAndCombine(coor, value, pred):
    # tensor to ndarray and post-processing
    coor1 = coor[0].detach().numpy()
    coor2 = coor[1].detach().numpy()
    value1 = value[0].detach().numpy()[0][0]
    value2 = value[1].detach().numpy()[0][0]
    value3 = value[2].detach().numpy()[0][0]
    value4 = value[3].detach().numpy()[0][0]
    predCD1 = pred[0].detach().numpy()[0][0]
    predCD2 = pred[1].detach().numpy()[0][0]
    predCD3 = pred[2].detach().numpy()[0][0]
    predCD4 = pred[3].detach().numpy()[0][0]
    predFUS1 = pred[4].detach().numpy()[0][0]
    predFUS2 = pred[5].detach().numpy()[0][0]
    predFUS3 = pred[6].detach().numpy()[0][0]
    predFUS4 = pred[7].detach().numpy()[0][0]
    x = coor1[0][0]
    y = coor1[0][1]
    x2 = coor2[0][0]
    y2 = coor2[0][1]
    dirichlet_boundary = [10, 7, 5, 1]  # up down left right
    value_min = min(dirichlet_boundary)
    value_max = max(dirichlet_boundary)

    # plotting
    fig, big_axes = plt.subplots(nrows = 4, ncols = 1, figsize = (9, 12), sharey = False, sharex = False)
    for i in range(4):
        big_axes[i].tick_params(labelcolor = (0., 0., 0., 0.), top = False, bottom = False, left = False,
                                right = False)
        big_axes[i]._frameon = False

    big_axes[0].set_title('(a) Data1', y = -0.2, fontsize = 15)
    big_axes[1].set_title('(b) Data2', y = -0.2, fontsize = 15)
    big_axes[2].set_title('(c) Data3', y = -0.2, fontsize = 15)
    big_axes[3].set_title('(d) Data4', y = -0.2, fontsize = 15)

    # Data1
    ax1 = fig.add_subplot(4, 3, 1)
    cf = ax1.scatter(x, y, c = value1, alpha = 0.9, edgecolors = 'none',
                     cmap = 'RdYlBu', marker = 's', s = 16, vmin = value_min, vmax = value_max)
    ax1.axis('square')
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    ax1.set_title('Truth')
    fig.colorbar(cf, ax = ax1)

    ax2 = fig.add_subplot(4, 3, 2)
    cf = ax2.scatter(x, y, c = predCD1, alpha = 0.9, edgecolors = 'none',
                     cmap = 'RdYlBu', marker = 's', s = 16, vmin = value_min, vmax = value_max)
    ax2.axis('square')
    ax2.set_xlim([0, 1])
    ax2.set_ylim([0, 1])
    ax2.set_title('CD')
    fig.colorbar(cf, ax = ax2)

    ax3 = fig.add_subplot(4, 3, 3)
    cf = ax3.scatter(x, y, c = predFUS1, alpha = 0.9, edgecolors = 'none',
                     cmap = 'RdYlBu', marker = 's', s = 16, vmin = value_min, vmax = value_max)
    ax3.axis('square')
    ax3.set_xlim([0, 1])
    ax3.set_ylim([0, 1])
    ax3.set_title('FUS')
    fig.colorbar(cf, ax = ax3)

    # Data2
    ax4 = fig.add_subplot(4, 3, 4)
    cf = ax4.scatter(x2, y2, c = value2, alpha = 0.9, edgecolors = 'none',
                     cmap = 'RdYlBu', marker = 's', s = 32, vmin = value_min, vmax = value_max)
    ax4.axis('square')
    ax4.set_xlim([0, 1])
    ax4.set_ylim([0, 1])
    ax4.set_title('Truth')
    fig.colorbar(cf, ax = ax4)

    ax5 = fig.add_subplot(4, 3, 5)
    cf = ax5.scatter(x2, y2, c = predCD2, alpha = 0.9, edgecolors = 'none',
                     cmap = 'RdYlBu', marker = 's', s = 32, vmin = value_min, vmax = value_max)
    ax5.axis('square')
    ax5.set_xlim([0, 1])
    ax5.set_ylim([0, 1])
    ax5.set_title('CD')
    fig.colorbar(cf, ax = ax5)

    ax6 = fig.add_subplot(4, 3, 6)
    cf = ax6.scatter(x2, y2, c = predFUS2, alpha = 0.9, edgecolors = 'none',
                     cmap = 'RdYlBu', marker = 's', s = 32, vmin = value_min, vmax = value_max)
    ax6.axis('square')
    ax6.set_xlim([0, 1])
    ax6.set_ylim([0, 1])
    ax6.set_title('FUS')
    fig.colorbar(cf, ax = ax6)

    # Data3
    ax7 = fig.add_subplot(4, 3, 7)
    cf = ax7.scatter(x, y, c = value3, alpha = 0.9, edgecolors = 'none',
                     cmap = 'RdYlBu', marker = 's', s = 16, vmin = value_min, vmax = value_max)
    ax7.axis('square')
    ax7.set_xlim([0, 1])
    ax7.set_ylim([0, 1])
    ax7.set_title('Truth')
    fig.colorbar(cf, ax = ax7)

    ax8 = fig.add_subplot(4, 3, 8)
    cf = ax8.scatter(x, y, c = predCD3, alpha = 0.9, edgecolors = 'none',
                     cmap = 'RdYlBu', marker = 's', s = 16, vmin = value_min, vmax = value_max)
    ax8.axis('square')
    ax8.set_xlim([0, 1])
    ax8.set_ylim([0, 1])
    ax8.set_title('CD')
    fig.colorbar(cf, ax = ax8)

    ax9 = fig.add_subplot(4, 3, 9)
    cf = ax9.scatter(x, y, c = predFUS3, alpha = 0.9, edgecolors = 'none',
                     cmap = 'RdYlBu', marker = 's', s = 16, vmin = value_min, vmax = value_max)
    ax9.axis('square')
    ax9.set_xlim([0, 1])
    ax9.set_ylim([0, 1])
    ax9.set_title('FUS')
    fig.colorbar(cf, ax = ax9)

    # Data4
    ax10 = fig.add_subplot(4, 3, 10)
    cf = ax10.scatter(x, y, c = value4, alpha = 0.9, edgecolors = 'none',
                      cmap = 'RdYlBu', marker = 's', s = 16, vmin = value_min, vmax = value_max)
    ax10.axis('square')
    ax10.set_xlim([0, 1])
    ax10.set_ylim([0, 1])
    ax10.set_title('Truth')
    fig.colorbar(cf, ax = ax10)

    ax11 = fig.add_subplot(4, 3, 11)
    cf = ax11.scatter(x, y, c = predCD4, alpha = 0.9, edgecolors = 'none',
                      cmap = 'RdYlBu', marker = 's', s = 16, vmin = value_min, vmax = value_max)
    ax11.axis('square')
    ax11.set_xlim([0, 1])
    ax11.set_ylim([0, 1])
    ax11.set_title('CD')
    fig.colorbar(cf, ax = ax11)

    ax12 = fig.add_subplot(4, 3, 12)
    cf = ax12.scatter(x, y, c = predFUS4, alpha = 0.9, edgecolors = 'none',
                      cmap = 'RdYlBu', marker = 's', s = 16, vmin = value_min, vmax = value_max)
    ax12.axis('square')
    ax12.set_xlim([0, 1])
    ax12.set_ylim([0, 1])
    ax12.set_title('FUS')
    fig.colorbar(cf, ax = ax12)

    # save and show
    fig.set_facecolor('w')
    plt.tight_layout()
    plt.savefig('./figures/result_1.pdf', dpi = 600, format = 'pdf')
    # plt.show()


if __name__ == '__main__':
    # load predictions
    coor1, value1, predCD1 = loadAndPred('./data/data3.txt', 101, './models/data3/CD/model_20000.pt')
    coor2, value2, predCD2 = loadAndPred('./data/data4.txt', 51, './models/data4/CD/model_20000.pt')
    _, value3, predCD3 = loadAndPred('./data/data2.txt', 101, './models/data2/CD/model_20000.pt')
    _, value4, predCD4 = loadAndPred('./data/data1.txt', 101, './models/data1/CD/model_20000.pt')
    _, _, predFUS1 = loadAndPred('./data/data3.txt', 101, './models/data3/FUS/model_20000.pt')
    _, _, predFUS2 = loadAndPred('./data/data4.txt', 51, './models/data4/FUS/model_20000.pt')
    _, _, predFUS3 = loadAndPred('./data/data2.txt', 101, './models/data2/FUS/model_20000.pt')
    _, _, predFUS4 = loadAndPred('./data/data1.txt', 101, './models/data1/FUS/model_20000.pt')

    # plotting
    coor = [coor1, coor2]
    value = [value1, value2, value3, value4]
    pred = [predCD1, predCD2, predCD3, predCD4, predFUS1, predFUS2, predFUS3, predFUS4]
    plottingAndCombine(coor, value, pred)
