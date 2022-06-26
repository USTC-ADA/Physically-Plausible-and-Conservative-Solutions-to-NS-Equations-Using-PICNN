# File      :_2Dsteady_NS_filter.py
# Time      :2022/3/16--15:57
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

# set random seeds and GPU environment
torch.manual_seed(42)
np.random.seed(42)
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def data_extraction(path, size = 101):
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
        self.conv_u1 = Conv2d(in_ch = 2, out_ch = 16, kernel = 5, stride = 1, pad = 2)
        self.conv_u2 = Conv2d(in_ch = 16, out_ch = 32, kernel = 5, stride = 1, pad = 2)
        self.conv_u3 = Conv2d(in_ch = 32, out_ch = 16, kernel = 5, stride = 1, pad = 2)
        self.conv_u31 = Conv2d(in_ch = 16, out_ch = 16, kernel = 5, stride = 1, pad = 2)
        self.conv_u4 = nn.Conv2d(in_channels = 16, out_channels = 1, kernel_size = 5, stride = 1, padding = 2)
        self.pixel_shuffle_u = nn.PixelShuffle(1)
        self.conv_v1 = Conv2d(in_ch = 2, out_ch = 16, kernel = 5, stride = 1, pad = 2)
        self.conv_v2 = Conv2d(in_ch = 16, out_ch = 32, kernel = 5, stride = 1, pad = 2)
        self.conv_v3 = Conv2d(in_ch = 32, out_ch = 16, kernel = 5, stride = 1, pad = 2)
        self.conv_v31 = Conv2d(in_ch = 16, out_ch = 16, kernel = 5, stride = 1, pad = 2)
        self.conv_v4 = nn.Conv2d(in_channels = 16, out_channels = 1, kernel_size = 5, stride = 1, padding = 2)
        self.pixel_shuffle_v = nn.PixelShuffle(1)
        self.conv_p1 = Conv2d(in_ch = 2, out_ch = 16, kernel = 5, stride = 1, pad = 2)
        self.conv_p2 = Conv2d(in_ch = 16, out_ch = 32, kernel = 5, stride = 1, pad = 2)
        self.conv_p3 = Conv2d(in_ch = 32, out_ch = 16, kernel = 5, stride = 1, pad = 2)
        self.conv_p31 = Conv2d(in_ch = 16, out_ch = 16, kernel = 5, stride = 1, pad = 2)
        self.conv_p4 = nn.Conv2d(in_channels = 16, out_channels = 1, kernel_size = 5, stride = 1, padding = 2)
        self.pixel_shuffle_p = nn.PixelShuffle(1)
        self._initialize_weights()

    def forward(self, x):
        x = self.upsample(x)

        u = self.conv_u1(x)
        u = self.conv_u2(u)
        u = self.conv_u3(u)
        # u = self.conv_u31(u)
        u = self.pixel_shuffle_u(self.conv_u4(u))

        v = self.conv_v1(x)
        v = self.conv_v2(v)
        v = self.conv_v3(v)
        # v = self.conv_v31(v)
        v = self.pixel_shuffle_v(self.conv_v4(v))

        p = self.conv_p1(x)
        p = self.conv_p2(p)
        p = self.conv_p3(p)
        # p = self.conv_p31(p)
        p = self.pixel_shuffle_p(self.conv_p4(p))
        return torch.cat((u, v, p), dim = 1)

    def _initialize_weights(self):
        init.kaiming_normal_(self.conv_u4.weight)
        init.kaiming_normal_(self.conv_v4.weight)
        init.kaiming_normal_(self.conv_p4.weight)


class Laplace(nn.Module):
    def __init__(self, delta_xy):
        super(Laplace, self).__init__()

        # CD scheme
        derFilter = [[[[0, 1, 0],
                       [1, -4, 1],
                       [0, 1, 0]]]]
        derFilter = torch.tensor(derFilter, dtype = torch.float32).to(device) / (delta_xy ** 2)
        self.filter = nn.Conv2d(in_channels = 1, out_channels = 1, kernel_size = 3, stride = 1, padding = 0,
                                bias = False)
        self.filter.weight = nn.Parameter(derFilter, requires_grad = False)

    def forward(self, x):
        derivate = self.filter(x)
        return derivate


class Partial_x(nn.Module):
    def __init__(self, delta_xy):
        super(Partial_x, self).__init__()

        # CD scheme
        derFilter = [[[[0, 0, 0],
                       [-1, 0, 1],
                       [0, 0, 0]]]]
        derFilter = torch.tensor(derFilter, dtype = torch.float32).to(device) / (delta_xy * 2)
        self.filter = nn.Conv2d(in_channels = 1, out_channels = 1, kernel_size = 3, stride = 1, padding = 0,
                                bias = False)
        self.filter.weight = nn.Parameter(derFilter, requires_grad = False)

    def forward(self, x):
        derivate = self.filter(x)
        return derivate


class Partial_y(nn.Module):
    def __init__(self, delta_xy):
        super(Partial_y, self).__init__()

        # CD scheme
        derFilter = [[[[0, -1, 0],
                       [0, 0, 0],
                       [0, 1, 0]]]]
        derFilter = torch.tensor(derFilter, dtype = torch.float32).to(device) / (delta_xy * 2)
        self.filter = nn.Conv2d(in_channels = 1, out_channels = 1, kernel_size = 3, stride = 1, padding = 0,
                                bias = False)
        self.filter.weight = nn.Parameter(derFilter, requires_grad = False)

    def forward(self, x):
        derivate = self.filter(x)
        return derivate


class Upwind_x(nn.Module):
    def __init__(self, delta_xy):
        super(Upwind_x, self).__init__()

        # upwind scheme
        derFilter = [[[[0, 0, 0],
                       [-1, 1, 0],
                       [0, 0, 0]]]]
        derFilter = torch.tensor(derFilter, dtype = torch.float32).to(device) / delta_xy
        self.filter = nn.Conv2d(in_channels = 1, out_channels = 1, kernel_size = 3, stride = 1, padding = 0,
                                bias = False)
        self.filter.weight = nn.Parameter(derFilter, requires_grad = False)

    def forward(self, x):
        derivate = self.filter(x)
        return derivate


class Upwind_y(nn.Module):
    def __init__(self, delta_xy):
        super(Upwind_y, self).__init__()

        # upwind scheme
        derFilter = [[[[0, -1, 0],
                       [0, 1, 0],
                       [0, 0, 0]]]]
        derFilter = torch.tensor(derFilter, dtype = torch.float32).to(device) / delta_xy
        self.filter = nn.Conv2d(in_channels = 1, out_channels = 1, kernel_size = 3, stride = 1, padding = 0,
                                bias = False)
        self.filter.weight = nn.Parameter(derFilter, requires_grad = False)

    def forward(self, x):
        derivate = self.filter(x)
        return derivate


class Upwind2_x(nn.Module):
    def __init__(self, delta_xy):
        super(Upwind2_x, self).__init__()

        # upwind scheme
        derFilter = [[[[0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0],
                       [1, -4, 3, 0, 0],
                       [0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0], ]]]
        derFilter = torch.tensor(derFilter, dtype = torch.float32).to(device) / (delta_xy * 2)
        self.filter = nn.Conv2d(in_channels = 1, out_channels = 1, kernel_size = 5, stride = 1, padding = 0,
                                bias = False)
        self.filter.weight = nn.Parameter(derFilter, requires_grad = False)

    def forward(self, x):
        derivate = self.filter(x)
        return derivate


class Upwind2_y(nn.Module):
    def __init__(self, delta_xy):
        super(Upwind2_y, self).__init__()

        # upwind scheme
        derFilter = [[[[0, 0, 1, 0, 0],
                       [0, 0, -4, 0, 0],
                       [0, 0, 3, 0, 0],
                       [0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0], ]]]
        derFilter = torch.tensor(derFilter, dtype = torch.float32).to(device) / (delta_xy * 2)
        self.filter = nn.Conv2d(in_channels = 1, out_channels = 1, kernel_size = 5, stride = 1, padding = 0,
                                bias = False)
        self.filter.weight = nn.Parameter(derFilter, requires_grad = False)

    def forward(self, x):
        derivate = self.filter(x)
        return derivate


def boundary_encoding(u, v, p, init_u, delta_xy, inlet, outlet):
    p[0, 0, 0, :] = p[0, 0, 1, :]
    p[0, 0, -1, :] = p[0, 0, -2, :]
    p[0, 0, :, 0] = p[0, 0, :, 1]
    p[0, 0, :, -1] = p[0, 0, :, -2]
    # up boundary is outlet, pressure on outlet needs to be reset
    for i in range(p.shape[3]):
        if outlet[0][0] < i * delta_xy < outlet[1][0]:
            p[0, 0, -1, i] = 0
        elif i * delta_xy == outlet[1][0]:
            p[0, 0, -1, i] = p[0, 0, -1, i + 1] / 2
        elif i * delta_xy == outlet[0][0]:
            p[0, 0, -1, i] = p[0, 0, -1, i - 1] / 2
    # left boundary is inlet, u on inlet needs to be reset
    for i in range(u.shape[2]):
        if inlet[0][1] < i * delta_xy < inlet[1][1]:
            u[0, 0, i, 0] = init_u
        elif i * delta_xy == inlet[0][1] or i * delta_xy == inlet[1][1]:
            u[0, 0, i, 0] = init_u / 2
    # up boundary is outlet, v on outlet needs to be reset
    for i in range(v.shape[3]):
        if outlet[0][0] <= i * delta_xy <= outlet[1][0]:
            v[0, 0, -1, i] = v[0, 0, -2, i]
    return u, v, p


def sus_boundary(u, v):
    constpad = nn.ConstantPad2d([1, 1, 1, 1], 0)
    u = constpad(u)
    v = constpad(v)
    u[0, 0, 1:-1, 0] = u[0, 0, 1:-1, 1]
    v[0, 0, -1, 1:-1] = v[0, 0, -2, 1:-1]
    return u, v


def phyloss(u, v, p, mu, rho, init_u, delta_xy, inlet, outlet, scheme):
    # Momentum Equation of u, v
    # boundary encoding of u, v, p
    u, v, p = boundary_encoding(u, v, p, init_u, delta_xy, inlet, outlet)
    # compute derivatives
    partial_x = Partial_x(delta_xy)
    partial_y = Partial_y(delta_xy)
    laplace = Laplace(delta_xy)
    upwind_x = Upwind_x(delta_xy)
    upwind_y = Upwind_y(delta_xy)

    dpdx = partial_x(p)
    dpdy = partial_y(p)
    laplace_u = laplace(u)
    laplace_v = laplace(v)
    dudx1 = partial_x(u)
    dvdy1 = partial_y(v)

    # u, v = sus_boundary(u, v)
    dudx = upwind_x(u)
    dudy = upwind_y(u)
    dvdx = upwind_x(v)
    dvdy = upwind_y(v)

    u = u[0, 0, 1:-1, 1:-1]
    v = v[0, 0, 1:-1, 1:-1]
    loss_u = rho * u * dudx + rho * v * dudy + dpdx - mu * laplace_u
    loss_v = rho * u * dvdx + rho * v * dvdy + dpdy - mu * laplace_v

    # Mass Equation
    mass = rho * (dudx1 + dvdy1)

    # compute mse loss
    mse_u = nn.MSELoss()
    mse_v = nn.MSELoss()
    mse_mass = nn.MSELoss()
    mse_u.to(device)
    mse_v.to(device)
    mse_mass.to(device)
    loss_u = mse_u(loss_u, torch.zeros_like(loss_u))
    loss_v = mse_v(loss_v, torch.zeros_like(loss_v))
    mass = mse_mass(mass, torch.zeros_like(mass))
    return loss_u, loss_v, mass


def frobenius_norm(tensor):
    return torch.sqrt(torch.sum(torch.pow(tensor, 2)))


def train(coor, value, model, epoch, mu, rho, init_u, delta_xy, inlet, outlet, scheme, lr = 0.001):
    optimizer = optim.Adam(model.parameters(), lr = lr)
    scheduler = StepLR(optimizer, step_size = 50000, gamma = 0.5)

    # load previous model
    # model, optimizer, scheduler = load_checkpoint(model, optimizer, scheduler)

    # optimizer = optim.Adam(model.parameters(), lr = 1e-5)
    # scheduler = StepLR(optimizer, step_size = 30000, gamma = 0.5)
    print(optimizer.state_dict()['param_groups'][0]['lr'])
    model.to(device)
    coor = coor.to(device)
    value = value.to(device)

    for i in range(epoch):
        uvp = model(coor)
        constpad = nn.ConstantPad2d([1, 1, 1, 1], 0)
        uvp_pad = constpad(uvp)
        u = uvp_pad[0, 0, :, :].unsqueeze(0).unsqueeze(0)
        v = uvp_pad[0, 1, :, :].unsqueeze(0).unsqueeze(0)
        p = uvp_pad[0, 2, :, :].unsqueeze(0).unsqueeze(0)
        loss_u, loss_v, mass = phyloss(u, v, p, mu, rho, init_u, delta_xy, inlet, outlet, scheme)
        loss = loss_u + loss_v + mass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        print('epoch:', i, 'loss:', loss)
        if (i + 1) % 100 == 0:
            save_checkpoint(model, optimizer, scheduler, save_dir = './models/model_' + str(i + 1) + '.pt')
            with open('./models/loss.txt', 'a+') as f:
                f.write(str(loss_u.item()) + ',' + str(loss_v.item()) + ',' + str(mass.item()) + '\n')
    # test
    pred_uvp = model(coor)
    constpad = nn.ConstantPad2d([1, 1, 1, 1], 0)
    uvp_pad = constpad(pred_uvp)
    u = uvp_pad[0, 0, :, :].unsqueeze(0).unsqueeze(0)
    v = uvp_pad[0, 1, :, :].unsqueeze(0).unsqueeze(0)
    p = uvp_pad[0, 2, :, :].unsqueeze(0).unsqueeze(0)
    u, v, p = boundary_encoding(u, v, p, init_u, delta_xy, inlet, outlet)
    error_u = frobenius_norm(u[0][0] - value[0][0]) / frobenius_norm(value[0][0])
    error_v = frobenius_norm(v[0][0] - value[0][1]) / frobenius_norm(value[0][1])
    error_p = frobenius_norm(p[0][0] - value[0][2]) / frobenius_norm(value[0][2])
    print('predict error_u:', error_u, '\npredict error_v:', error_v, '\npredict error_p:', error_p)
    # print(pred)
    return {error_u, error_v, error_p}


def save_checkpoint(model, optimizer, scheduler, save_dir = './models/model_3000.pt'):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
    }, save_dir)


def load_checkpoint(model, optimizer, scheduler, save_dir = './models/model_126500.pt'):
    checkpoint = torch.load(save_dir)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    print('Pretrained model loaded!')

    return model, optimizer, scheduler


def test(coor, value, model, delta_xy, init_u, inlet, outlet, path):
    # load model
    optimizer = optim.Adam(model.parameters(), lr = 0.01)
    scheduler = StepLR(optimizer, step_size = 500, gamma = 0.97)
    model, _, _ = load_checkpoint(model, optimizer, scheduler, path)

    # test
    pred_uvp = model(coor)
    constpad = nn.ConstantPad2d([1, 1, 1, 1], 0)
    uvp_pad = constpad(pred_uvp)
    u = uvp_pad[0, 0, :, :].unsqueeze(0).unsqueeze(0)
    v = uvp_pad[0, 1, :, :].unsqueeze(0).unsqueeze(0)
    p = uvp_pad[0, 2, :, :].unsqueeze(0).unsqueeze(0)
    u, v, p = boundary_encoding(u, v, p, init_u, delta_xy, inlet, outlet)

    error_u = frobenius_norm(u[0][0] - value[0][0]) / frobenius_norm(value[0][0])
    error_v = frobenius_norm(v[0][0] - value[0][1]) / frobenius_norm(value[0][1])
    error_p = frobenius_norm(p[0][0] - value[0][2]) / frobenius_norm(value[0][2])

    uv_pred = torch.pow((torch.pow(u[0][0], 2) + torch.pow(v[0][0], 2)), 0.5)
    uv_value = torch.pow((torch.pow(value[0][0], 2) + torch.pow(value[0][1], 2)), 0.5)
    error_uv = frobenius_norm(uv_pred - uv_value) / frobenius_norm(uv_value)
    print('predict error_u:', error_u, '\npredict error_v:', error_v, '\npredict error_p:', error_p,
          '\npredict error_uv:', error_uv)
    return torch.cat((u, v, p), dim = 1)


def plotting(coor, value, pred, fig_path):
    # tensor to ndarray and post-processing
    coor = coor.cpu().detach().numpy()
    value_u = value.cpu().detach().numpy()[0][0]
    value_v = value.cpu().detach().numpy()[0][1]
    value_p = value.cpu().detach().numpy()[0][2]
    pred_u = pred.cpu().detach().numpy()[0][0]
    pred_v = pred.cpu().detach().numpy()[0][1]
    pred_p = pred.cpu().detach().numpy()[0][2]
    value_uv = np.power(np.power(value_u, 2) + np.power(value_v, 2), 0.5)
    pred_uv = np.power(np.power(pred_u, 2) + np.power(pred_v, 2), 0.5)
    x = coor[0][0]
    y = coor[0][1]
    uv_min = min(value_uv.flatten())
    uv_max = max(value_uv.flatten())
    p_min = min(value_p.flatten())
    p_max = max(value_p.flatten())
    print(max(value_p.flatten()))
    print(max(pred_p.flatten()))
    print(min(value_p.flatten()))
    print(min(pred_p.flatten()))

    # plotting
    fig, ax = plt.subplots(nrows = 2, ncols = 2, figsize = (7, 7))
    fig.subplots_adjust(hspace = 0.3, wspace = 0.3)

    cf = ax[0, 0].scatter(x, y, c = value_uv, alpha = 0.9, edgecolors = 'none',
                          cmap = 'coolwarm', marker = 's', s = 16, vmin = uv_min, vmax = uv_max)  # RdYlBu
    ax[0, 0].axis('square')
    ax[0, 0].set_xlim([0, 1])
    ax[0, 0].set_ylim([0, 1])
    ax[0, 0].set_title('Ref velocity')
    fig.colorbar(cf, ax = ax[0, 0])

    cf = ax[0, 1].scatter(x, y, c = pred_uv, alpha = 0.9, edgecolors = 'none',
                          cmap = 'coolwarm', marker = 's', s = 16, vmin = uv_min, vmax = uv_max)
    ax[0, 1].axis('square')
    ax[0, 1].set_xlim([0, 1])
    ax[0, 1].set_ylim([0, 1])
    ax[0, 1].set_title('pred velocity')
    fig.colorbar(cf, ax = ax[0, 1])

    cf = ax[1, 0].scatter(x, y, c = value_p, alpha = 0.9, edgecolors = 'none',
                          cmap = 'coolwarm', marker = 's', s = 16, vmin = p_min, vmax = p_max)
    ax[1, 0].axis('square')
    ax[1, 0].set_xlim([0, 1])
    ax[1, 0].set_ylim([0, 1])
    ax[1, 0].set_title('Ref p')
    fig.colorbar(cf, ax = ax[1, 0])

    cf = ax[1, 1].scatter(x, y, c = pred_p, alpha = 0.9, edgecolors = 'none',
                          cmap = 'coolwarm', marker = 's', s = 16, vmin = p_min, vmax = p_max)
    ax[1, 1].axis('square')
    ax[1, 1].set_xlim([0, 1])
    ax[1, 1].set_ylim([0, 1])
    ax[1, 1].set_title('pred p')
    fig.colorbar(cf, ax = ax[1, 1])

    # save and show
    # plt.savefig(fig_path)
    plt.show()


if __name__ == '__main__':
    # set parameters
    data_path = './data/data4.txt'
    mu = 0.01
    rho = 1
    init_u = 1
    delta_xy = 0.02
    inlet = [[0, 0.2], [0, 0.5]]
    outlet = [[0.4, 1], [0.7, 1]]
    epoch = 200000
    data_size = 51
    scheme = 'SUS'

    # load data
    coor, value = data_extraction(data_path, data_size)
    coor = torch.tensor(coor, dtype = torch.float32).resize_((1, coor.shape[0], coor.shape[1], coor.shape[2]))
    value = torch.tensor(value, dtype = torch.float32).resize_((1, value.shape[0], value.shape[1], value.shape[2]))

    # set model and train
    model = PICNN(upsample_size = [data_size - 2, data_size - 2])

    # start_time = time.time()
    # error = train(coor, value, model, epoch, mu, rho, init_u, delta_xy, inlet, outlet, scheme)
    # NStime = time.time() - start_time

    # print error
    # print('error:', error, 'time:', NStime)

    # test
    pred = test(coor, value, model, delta_xy, init_u, inlet, outlet, './models/data4_SUSf/model_200000.pt')

    # plotting
    # plotting(coor, value, pred, './figures/result.png')
