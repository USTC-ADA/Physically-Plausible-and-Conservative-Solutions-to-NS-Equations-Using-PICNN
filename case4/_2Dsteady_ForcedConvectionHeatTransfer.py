# File      :_2Dsteady_ForcedConvectionHeatTransfer.py
# Time      :2022/3/27--21:30
# Author    :JF Li
# Version   :python 3.7

import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn.init as init
import matplotlib.pyplot as plt
import re
import os
import sys

sys.path.insert(0, '../case3')
import _2Dsteady_NS

# set random seeds and GPU environment
torch.manual_seed(42)
np.random.seed(42)
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def data_extraction(path, size = 51):
    data = []
    with open(path, 'r', encoding = 'utf-8') as file:
        while True:
            line = file.readline()
            if not line:
                break
            if line[0] != '%':
                line = re.sub(' +', ' ', line)
                strline = line.split(' ', 5)
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
    value = np.zeros((1, size, size), dtype = float)
    visit = np.zeros((size, size), dtype = float)
    valid = 0
    try:
        for i in range(data.shape[0]):
            index_x = int(data[i][0] * (size - 1) + 0.1)
            index_y = int(data[i][1] * (size - 1) + 0.1)
            if visit[index_y][index_x] != 0:
                raise Exception("set real value error!")
            else:
                value[0][index_y][index_x] = data[i][5]
                visit[index_y][index_x] = 1
                valid += 1
    except Exception as e:
        print(e)
    assert valid == (size * size), "size error!"
    return coor, value


def get_velocity(coor, delta_xy, init_u, inlet, outlet, data_size, path):
    # load model
    model = _2Dsteady_NS.PICNN(upsample_size = [data_size - 2, data_size - 2])
    optimizer = optim.Adam(model.parameters(), lr = 0.01)
    scheduler = StepLR(optimizer, step_size = 500, gamma = 0.97)
    model, _, _ = load_checkpoint(model, optimizer, scheduler, path)

    # get uv
    pred_uvp = model(coor)
    constpad = nn.ConstantPad2d([1, 1, 1, 1], 0)
    uvp_pad = constpad(pred_uvp)
    u = uvp_pad[0, 0, :, :].unsqueeze(0).unsqueeze(0)
    v = uvp_pad[0, 1, :, :].unsqueeze(0).unsqueeze(0)
    p = uvp_pad[0, 2, :, :].unsqueeze(0).unsqueeze(0)
    u, v, _ = _2Dsteady_NS.boundary_encoding(u, v, p, init_u, delta_xy, inlet, outlet)
    print('velocity loaded!')
    return torch.cat((u, v), dim = 1)


class Conv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, stride, pad):
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels = in_ch, out_channels = out_ch, kernel_size = kernel, stride = stride,
                              padding = pad)
        self.relu = nn.LeakyReLU()
        self._initialize_weights()

    def forward(self, x):
        return self.relu(self.conv(x))

    def _initialize_weights(self):
        init.kaiming_normal_(self.conv.weight, mode = 'fan_out', nonlinearity = 'relu')


class PICNN(nn.Module):
    def __init__(self, upsample_size):
        super(PICNN, self).__init__()
        self.upsample = nn.Upsample(size = upsample_size, mode = 'bilinear', align_corners = True)
        self.conv1 = Conv2d(in_ch = 2, out_ch = 16, kernel = 5, stride = 1, pad = 2)
        self.conv2 = Conv2d(in_ch = 16, out_ch = 32, kernel = 5, stride = 1, pad = 2)
        self.conv3 = Conv2d(in_ch = 32, out_ch = 16, kernel = 5, stride = 1, pad = 2)
        self.conv31 = Conv2d(in_ch = 16, out_ch = 16, kernel = 5, stride = 1, pad = 2)
        self.conv4 = nn.Conv2d(in_channels = 16, out_channels = 1, kernel_size = 5, stride = 1, padding = 2)
        self.pixel_shuffle = nn.PixelShuffle(1)
        self._initialize_weights()

    def forward(self, x):
        out = self.upsample(x)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        # out = self.conv31(out)
        out = self.pixel_shuffle(self.conv4(out))
        return out

    def _initialize_weights(self):
        init.kaiming_normal_(self.conv4.weight)


def A(p_delta, scheme):
    # compute the function A(P_delta)
    if scheme == 'CD':
        return 1 - 0.5 * p_delta
    elif scheme == 'FUS' or scheme == 'SUS':
        return torch.ones_like(p_delta).to(device)
    elif scheme == 'HS':
        return max(0, 1 - 0.5 * p_delta)
    elif scheme == 'ES':
        return p_delta / (np.exp(p_delta) - 1)
    elif scheme == 'PLS':
        r = torch.pow(1 - 0.1 * p_delta, 5)
        r1 = torch.max(torch.cat((r.unsqueeze(0), torch.zeros_like(r.unsqueeze(0)).to(device)), dim = 0), dim = 0)[0]
        return r1
    else:
        return 0


def boundary_encoding(temperature, delta_xy, inlet, dirichlet_boundary):
    temperature[0, 0, 0, :] = dirichlet_boundary[1]  # 下
    temperature[0, 0, :, -1] = dirichlet_boundary[1]  # 右
    temperature[0, 0, -1, :] = dirichlet_boundary[1]  # 上
    temperature[0, 0, :, 0] = dirichlet_boundary[0]  # 左

    # left boundary is inlet, T on inlet needs to be reset
    # for i in range(temperature.shape[2]):
    #     if inlet[0][1] <= i * delta_xy <= inlet[1][1]:
    #         temperature[0, 0, i, 0] = dirichlet_boundary[0]
    return temperature


def sus_boundary(temperature):
    constpad = nn.ConstantPad2d([1, 1, 1, 1], 0)
    temperature = constpad(temperature)
    temperature[0, 0, :, 0] = temperature[0, 0, :, 1]
    temperature[0, 0, -1, :] = temperature[0, 0, -2, :]
    temperature[0, 0, 0, :] = temperature[0, 0, 1, :]
    temperature[0, 0, :, -1] = temperature[0, 0, :, -2]
    return temperature


def phyloss(temperature, gamma, delta_xy, inlet, dirichlet_boundary, uv, scheme = 'FUS'):
    loss = torch.zeros(temperature.shape[2] - 2, temperature.shape[3] - 2).to(device)

    # boundary encoding of T
    temperature = boundary_encoding(temperature, delta_xy, inlet, dirichlet_boundary)
    u = uv[0][0].unsqueeze(0).unsqueeze(0)
    v = uv[0][1].unsqueeze(0).unsqueeze(0)

    ave_u = (u[0, 0, 1:-1, :-1] + u[0, 0, 1:-1, 1:]) / 2
    ave_v = (v[0, 0, :-1, 1:-1] + v[0, 0, 1:, 1:-1]) / 2
    p_delta_u = ave_u * delta_xy / gamma
    p_delta_v = ave_v * delta_xy / gamma
    max_N = torch.max(
        torch.cat((-1 * p_delta_v.unsqueeze(0), torch.zeros_like(p_delta_v.unsqueeze(0)).to(device)), dim = 0),
        dim = 0)[0]
    max_S = torch.max(
        torch.cat((p_delta_v.unsqueeze(0), torch.zeros_like(p_delta_v.unsqueeze(0)).to(device)), dim = 0),
        dim = 0)[0]
    max_W = torch.max(
        torch.cat((p_delta_u.unsqueeze(0), torch.zeros_like(p_delta_u.unsqueeze(0)).to(device)), dim = 0),
        dim = 0)[0]
    max_E = torch.max(
        torch.cat((-1 * p_delta_u.unsqueeze(0), torch.zeros_like(p_delta_u.unsqueeze(0)).to(device)), dim = 0),
        dim = 0)[0]
    a_N = (A(torch.abs(p_delta_v), scheme = scheme) + max_N)
    a_S = (A(torch.abs(p_delta_v), scheme = scheme) + max_S)
    a_W = (A(torch.abs(p_delta_u), scheme = scheme) + max_W)
    a_E = (A(torch.abs(p_delta_u), scheme = scheme) + max_E)
    loss += a_N[1:, :] * (temperature[0, 0, 2:, 1:-1] - temperature[0, 0, 1:-1, 1:-1]) - \
            p_delta_v[1:, :] * temperature[0, 0, 1:-1, 1:-1]
    loss += a_S[:-1, :] * (temperature[0, 0, :-2, 1:-1] - temperature[0, 0, 1:-1, 1:-1]) + \
            p_delta_v[:-1, :] * temperature[0, 0, 1:-1, 1:-1]
    loss += a_W[:, :-1] * (temperature[0, 0, 1:-1, :-2] - temperature[0, 0, 1:-1, 1:-1]) + \
            p_delta_u[:, :-1] * temperature[0, 0, 1:-1, 1:-1]
    loss += a_E[:, 1:] * (temperature[0, 0, 1:-1, 2:] - temperature[0, 0, 1:-1, 1:-1]) - \
            p_delta_u[:, 1:] * temperature[0, 0, 1:-1, 1:-1]

    # compute sus
    if scheme == 'SUS':
        temperature = sus_boundary(temperature)
        loss += (temperature[0, 0, 2:-2, 1:-3] * (max_W[:, 1:] + max_W[:, :-1]) +
                 temperature[0, 0, 2:-2, 3:-1] * (max_E[:, 1:] + max_E[:, :-1]) +
                 temperature[0, 0, 1:-3, 2:-2] * (max_S[1:, :] + max_S[:-1, :]) +
                 temperature[0, 0, 3:-1, 2:-2] * (max_N[1:, :] + max_N[:-1, :]) -
                 temperature[0, 0, 2:-2, 2:-2] * (max_W[:, 1:] + max_E[:, :-1] + max_S[1:, :] + max_N[:-1, :]) -
                 temperature[0, 0, 2:-2, :-4] * max_W[:, :-1] -
                 temperature[0, 0, 2:-2, 4:] * max_E[:, 1:] -
                 temperature[0, 0, :-4, 2:-2] * max_S[:-1, :] -
                 temperature[0, 0, 4:, 2:-2] * max_N[1:, :]) * 0.5

    # compute mse loss
    mse_T = nn.MSELoss()
    mse_T.to(device)
    loss = mse_T(loss, torch.zeros_like(loss))
    return loss


def frobenius_norm(tensor):
    return torch.sqrt(torch.sum(torch.pow(tensor, 2)))


def train(coor, value, model, epoch, gamma, delta_xy, inlet, dirichlet_boundary, uv, scheme = 'FUS', lr = 0.001):
    optimizer = optim.Adam(model.parameters(), lr = lr)
    scheduler = StepLR(optimizer, step_size = 50000, gamma = 0.5)

    # load previous model
    # model, optimizer, scheduler = load_checkpoint(model, optimizer, scheduler,
    #                                               save_dir = './models/model_200000.pt')

    # optimizer = optim.Adam(model.parameters(), lr = 1.25e-4)
    # scheduler = StepLR(optimizer, step_size = 5000, gamma = 0.5)
    print(optimizer.state_dict()['param_groups'][0]['lr'])
    model.to(device)
    coor = coor.to(device)
    value = value.to(device)
    uv = uv.cpu().detach()
    uv = uv.to(device)

    for i in range(epoch):
        temperature = model(coor)
        constpad = nn.ConstantPad2d([1, 1, 1, 1], 0)
        temperature_pad = constpad(temperature)
        loss = phyloss(temperature_pad, gamma, delta_xy, inlet, dirichlet_boundary, uv, scheme)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        print('epoch:', i, 'loss:', loss)
        if (i + 1) % 100 == 0:
            save_checkpoint(model, optimizer, scheduler, save_dir = './models/model_' + str(i + 1) + '.pt')
            with open('./models/loss.txt', 'a+') as f:
                f.write(str(loss.item()) + '\n')
    # test
    pred_T = model(coor)
    constpad = nn.ConstantPad2d([1, 1, 1, 1], 0)
    T_pad = constpad(pred_T)
    temperature = boundary_encoding(T_pad, delta_xy, inlet, dirichlet_boundary)
    error_T = frobenius_norm((temperature[0][0] + 273.15) - value[0][0]) / frobenius_norm(value[0][0])
    print('predict error_T:', error_T)
    # print(pred)
    return error_T


def save_checkpoint(model, optimizer, scheduler, save_dir = './models/model_100.pt'):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
    }, save_dir)


def load_checkpoint(model, optimizer, scheduler, save_dir = './models/model_100.pt'):
    checkpoint = torch.load(save_dir)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    print('Pretrained model loaded!')

    return model, optimizer, scheduler


def test(coor, value, model, delta_xy, inlet, dirichlet_boundary, path):
    # load model
    optimizer = optim.Adam(model.parameters(), lr = 0.01)
    scheduler = StepLR(optimizer, step_size = 500, gamma = 0.97)
    model, _, _ = load_checkpoint(model, optimizer, scheduler, path)

    # test
    pred_T = model(coor)
    constpad = nn.ConstantPad2d([1, 1, 1, 1], 0)
    T_pad = constpad(pred_T)
    temperature = boundary_encoding(T_pad, delta_xy, inlet, dirichlet_boundary)

    error_T = frobenius_norm((temperature[0][0] + 273.15) - value[0][0]) / frobenius_norm(value[0][0])
    print('predict error_T:', error_T)
    return temperature


def plotting(coor, value, pred, fig_path):
    # tensor to ndarray and post-processing
    coor = coor.cpu().detach().numpy()
    value = value.cpu().detach().numpy()[0][0] - 273.15
    pred = pred.cpu().detach().numpy()[0][0]
    x = coor[0][0]
    y = coor[0][1]
    t_min = min(value.flatten())
    t_max = max(value.flatten())
    t_min1 = min(pred.flatten())
    t_max1 = max(pred.flatten())
    print(t_min)
    print(t_max)
    print(t_min1)
    print(t_max1)

    # plotting
    fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (7, 7))
    fig.subplots_adjust(hspace = 0.3, wspace = 0.3)

    cf = ax[0].scatter(x, y, c = value, alpha = 0.9, edgecolors = 'none',
                       cmap = 'rainbow', marker = 's', s = 16, vmin = t_min, vmax = t_max)  # RdYlBu coolwarm
    ax[0].axis('square')
    ax[0].set_xlim([0, 1])
    ax[0].set_ylim([0, 1])
    ax[0].set_title('Ref temperature')
    fig.colorbar(cf, ax = ax[0])

    cf = ax[1].scatter(x, y, c = pred, alpha = 0.9, edgecolors = 'none',
                       cmap = 'rainbow', marker = 's', s = 16, vmin = t_min, vmax = t_max)
    ax[1].axis('square')
    ax[1].set_xlim([0, 1])
    ax[1].set_ylim([0, 1])
    ax[1].set_title('pred temperature')
    fig.colorbar(cf, ax = ax[1])

    # save and show
    # plt.savefig(fig_path)
    plt.show()


if __name__ == '__main__':
    # set parameters
    data_path = './data/data10.txt'
    # uv_path = '../case3/models/data8_SUS/model_200000.pt'
    gamma = 0.01
    delta_xy = 0.02
    dirichlet_boundary = [20, 60]  # inlet, up
    inlet = [[0, 0.2], [0, 0.7]]
    outlet = [[1, 0.4], [1, 0.9]]
    init_u = 1
    epoch = 200000
    data_size = 51
    scheme = 'CD'

    # load data
    coor, value = data_extraction(data_path, data_size)
    coor = torch.tensor(coor, dtype = torch.float32).resize_((1, coor.shape[0], coor.shape[1], coor.shape[2]))
    value = torch.tensor(value, dtype = torch.float32).resize_((1, value.shape[0], value.shape[1], value.shape[2]))
    # uv = get_velocity(coor, delta_xy, init_u, inlet, outlet, data_size, uv_path)
    # real value
    _, real = _2Dsteady_NS.data_extraction('../case4/data/data7_1.txt', data_size)
    real_uv = torch.tensor(real, dtype = torch.float32)[:2]
    real_uv = real_uv.resize_((1, real_uv.shape[0], real_uv.shape[1], real_uv.shape[2]))

    # set model and train
    model = PICNN(upsample_size = [data_size - 2, data_size - 2])

    # start_time = time.time()
    # error = train(coor, value, model, epoch, gamma, delta_xy, inlet, dirichlet_boundary, real_uv, scheme = scheme)
    # totaltime = time.time() - start_time

    # print error
    # print('error:', error, 'time:', totaltime)

    # test
    pred_T = test(coor, value, model, delta_xy, inlet, dirichlet_boundary, './models/data10_CD/model_200000.pt')

    # plotting
    plotting(coor, value, pred_T, './figures/result.png')
