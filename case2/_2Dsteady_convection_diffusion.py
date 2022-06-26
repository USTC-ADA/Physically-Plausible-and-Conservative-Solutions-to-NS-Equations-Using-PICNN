# File      :_2Dsteady_convection_diffusion.py
# Time      :2022/1/11--17:03
# Author    :JF Li
# Version   :python 3.7

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import re
import os

# set random seeds and GPU environment
torch.manual_seed(42)
np.random.seed(42)
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
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
                strline = line.split(' ', 2)
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
                value[0][index_y][index_x] = data[i][2]
                visit[index_y][index_x] = 1
                valid += 1
    except Exception as e:
        print(e)
    assert valid == (size * size), "size error!"
    return coor, value


class Conv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, stride, pad):
        super(Conv2d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels = in_ch, out_channels = out_ch, kernel_size = kernel, stride = stride, padding = pad),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace = True)
        )

    def forward(self, x):
        return self.conv(x)


class PICNN(nn.Module):
    def __init__(self, upsample_size):
        super(PICNN, self).__init__()
        self.conv1 = Conv2d(in_ch = 2, out_ch = 4, kernel = 5, stride = 1, pad = 1)
        self.conv2 = Conv2d(in_ch = 4, out_ch = 8, kernel = 5, stride = 1, pad = 1)
        self.conv3 = Conv2d(in_ch = 8, out_ch = 16, kernel = 5, stride = 1, pad = 1)
        self.upsample = nn.Upsample(size = [upsample_size, upsample_size])
        self.conv4 = nn.Conv2d(in_channels = 16, out_channels = 1, kernel_size = 5, stride = 1, padding = 2)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.upsample(out)
        out = self.conv4(out)
        return out


def A(p_delta, scheme):
    # compute the function A(P_delta)
    if scheme == 'CD':
        return 1 - 0.5 * p_delta
    elif scheme == 'FUS' or scheme == 'SUS':
        return 1
    elif scheme == 'HS':
        return max(0, 1 - 0.5 * p_delta)
    elif scheme == 'ES':
        return p_delta / (np.exp(p_delta) - 1)
    elif scheme == 'PLS':
        return max(0, np.power(1 - 0.1 * p_delta, 5))
    else:
        return 0


def compute_coefficient(F_x, F_y, gamma, delta_xy, scheme):
    # compute the low level format coefficients: a_E a_W a_N a_S a_P
    p_delta_lr = F_x * delta_xy / gamma
    p_delta_ud = F_y * delta_xy / gamma
    a_E = A(abs(p_delta_lr), scheme = scheme) + max(-p_delta_lr, 0)
    a_W = A(abs(p_delta_lr), scheme = scheme) + max(p_delta_lr, 0)
    a_N = A(abs(p_delta_ud), scheme = scheme) + max(-p_delta_ud, 0)
    a_S = A(abs(p_delta_ud), scheme = scheme) + max(p_delta_ud, 0)
    a_P = a_E + a_W + a_N + a_S
    derivative_filter = [[[[0, a_S, 0],
                           [a_W, -a_P, a_E],
                           [0, a_N, 0]]]]

    return derivative_filter


def compute_coefficient_sus(F_x, F_y, gamma, delta_xy):
    # compute the high level format coefficients
    p_delta_lr = F_x * delta_xy * 0.5 / gamma
    p_delta_ud = F_y * delta_xy * 0.5 / gamma
    a_u = max(p_delta_lr, 0)
    a_v = max(p_delta_ud, 0)
    a_P = a_u + a_v
    a_W = 2 * a_u
    a_S = 2 * a_v
    a_WW = a_u
    a_SS = a_v
    derivative_filter = [[[[0, 0, -a_SS, 0, 0],
                           [0, 0, a_S, 0, 0],
                           [-a_WW, a_W, -a_P, 0, 0],
                           [0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0], ]]]

    return derivative_filter


class Conv2dDerivative(nn.Module):
    def __init__(self, F_x, F_y, gamma, delta_xy, scheme):
        super(Conv2dDerivative, self).__init__()

        # FVM scheme
        derFilter = compute_coefficient(F_x, F_y, gamma, delta_xy, scheme = scheme)
        derFilter = torch.tensor(derFilter, dtype = torch.float32).to(device)
        self.filter = nn.Conv2d(in_channels = 1, out_channels = 1, kernel_size = 3, stride = 1, padding = 0,
                                bias = False)
        self.filter.weight = nn.Parameter(derFilter, requires_grad = False)

    def forward(self, x):
        derivate = self.filter(x)
        return derivate


class Conv2dDerivativeSUS(nn.Module):
    def __init__(self, F_x, F_y, gamma, delta_xy):
        super(Conv2dDerivativeSUS, self).__init__()

        # FVM scheme
        derFilter = compute_coefficient_sus(F_x, F_y, gamma, delta_xy)
        derFilter = torch.tensor(derFilter, dtype = torch.float32).to(device)
        self.filter = nn.Conv2d(in_channels = 1, out_channels = 1, kernel_size = 5, stride = 1, padding = 0,
                                bias = False)
        self.filter.weight = nn.Parameter(derFilter, requires_grad = False)

    def forward(self, x):
        derivate = self.filter(x)
        return derivate


def boundary_encoding(x, dirichlet_boundary):
    constpad = nn.ConstantPad2d([1, 1, 1, 1], 0)
    x_pad = constpad(x)
    # up down left right
    x_pad[0, 0, 0, :] = dirichlet_boundary[1]  # 下
    x_pad[0, 0, :, -1] = dirichlet_boundary[3]  # 右
    x_pad[0, 0, -1, :] = dirichlet_boundary[0]  # 上
    x_pad[0, 0, :, 0] = dirichlet_boundary[2]  # 左
    return x_pad


def sus(x, F_x, F_y, gamma, delta_xy, dirichlet_boundary):
    x = boundary_encoding(x, dirichlet_boundary)
    loss_func = Conv2dDerivativeSUS(F_x, F_y, gamma, delta_xy)
    loss = loss_func(x)
    mse_loss = nn.MSELoss()
    mse_loss.to(device)
    loss = mse_loss(loss, torch.zeros_like(loss))
    return loss


def phyloss(x, F_x, F_y, gamma, delta_xy, dirichlet_boundary, scheme):
    # encode boundary conditions
    x = boundary_encoding(x, dirichlet_boundary)

    # compute mse loss
    loss_func = Conv2dDerivative(F_x, F_y, gamma, delta_xy, scheme)
    loss = loss_func(x)
    mse_loss = nn.MSELoss()
    mse_loss.to(device)
    loss = mse_loss(loss, torch.zeros_like(loss))
    if scheme == 'SUS':
        loss_sus = sus(x, F_x, F_y, gamma, delta_xy, dirichlet_boundary)
        loss += loss_sus
    return loss


def frobenius_norm(tensor):
    return torch.sqrt(torch.sum(torch.pow(tensor, 2)))


def train(coor, value, model, epoch, F_x, F_y, gamma, delta_xy, dirichlet_boundary, scheme, lr = 0.001):
    optimizer = optim.Adam(model.parameters(), lr = lr)

    # load previous model
    # model = load_checkpoint(model)

    model.to(device)
    coor = coor.to(device)
    value = value.to(device)

    for i in range(epoch):
        output = model(coor)
        loss = phyloss(output, F_x, F_y, gamma, delta_xy, dirichlet_boundary, scheme)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('epoch:', i, 'loss:', loss)
        if (i + 1) % 100 == 0:
            save_checkpoint(model, save_dir = './models/model_' + str(i + 1) + '.pt')
            with open('./models/loss.txt', 'a+') as f:
                f.write(str(loss.item()) + '\n')
    # test
    pred = model(coor)
    pred = boundary_encoding(pred, dirichlet_boundary)
    error = frobenius_norm(pred - value) / frobenius_norm(value)
    print('predict error:', error)
    # print(pred)
    return error


def test(coor, value, model, dirichlet_boundary, path):
    # load model
    model = load_checkpoint(model, path)
    # test
    pred = model(coor)
    pred = boundary_encoding(pred, dirichlet_boundary)
    error = frobenius_norm(pred - value) / frobenius_norm(value)
    print('predict error:', error)
    return pred


def plotting(coor, value, pred, dirichlet_boundary, scheme, fig_path):
    # tensor to ndarray and post-processing
    coor = coor.detach().numpy()
    value = value.detach().numpy()[0][0]
    pred = pred.detach().numpy()[0][0]
    x = coor[0][0]
    y = coor[0][1]
    value_min = min(dirichlet_boundary)
    value_max = max(dirichlet_boundary)

    # plotting
    fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (7, 7))
    fig.subplots_adjust(hspace = 0.3, wspace = 0.3)

    cf = ax[0].scatter(x, y, c = value, alpha = 0.9, edgecolors = 'none',
                       cmap = 'RdYlBu', marker = 's', s = 16, vmin = value_min, vmax = value_max)
    ax[0].axis('square')
    ax[0].set_xlim([0, 1])
    ax[0].set_ylim([0, 1])
    ax[0].set_title('Ref')
    fig.colorbar(cf, ax = ax[0])

    cf = ax[1].scatter(x, y, c = pred, alpha = 0.9, edgecolors = 'none',
                       cmap = 'RdYlBu', marker = 's', s = 16, vmin = value_min, vmax = value_max)
    ax[1].axis('square')
    ax[1].set_xlim([0, 1])
    ax[1].set_ylim([0, 1])
    ax[1].set_title(scheme)
    fig.colorbar(cf, ax = ax[1])

    # save and show
    # plt.savefig(fig_path)
    plt.show()


def save_checkpoint(model, save_dir = './model/model_5000.pt'):
    torch.save({
        'model_state_dict': model.state_dict()
    }, save_dir)


def load_checkpoint(model, save_dir = './model/model_4000.pt'):
    checkpoint = torch.load(save_dir)
    model.load_state_dict(checkpoint['model_state_dict'])

    print('Pretrained model loaded!')

    return model


if __name__ == '__main__':
    # set parameters
    data_path = './data/data4.txt'
    gamma = 0.01
    rho_u = 1
    rho_v = 2
    delta_xy = 0.02
    dirichlet_boundary = [10, 7, 5, 1]  # up down left right
    epoch = 40000
    data_size = 51
    scheme = 'CD'

    # load data
    coor, value = data_extraction(data_path, data_size)
    coor = torch.tensor(coor, dtype = torch.float32).resize_((1, coor.shape[0], coor.shape[1], coor.shape[2]))
    value = torch.tensor(value, dtype = torch.float32).resize_((1, value.shape[0], value.shape[1], value.shape[2]))

    # set model and train
    model = PICNN(upsample_size = data_size - 2)
    # error = train(coor, value, model, epoch, rho_u, rho_v, gamma, delta_xy, dirichlet_boundary, scheme)

    # print error
    # print('error:', error)

    # test
    pred = test(coor, value, model, dirichlet_boundary, './models/data4/CD/model_20000.pt')

    # plotting
    plotting(coor, value, pred, dirichlet_boundary, scheme, './figures/result.png')
