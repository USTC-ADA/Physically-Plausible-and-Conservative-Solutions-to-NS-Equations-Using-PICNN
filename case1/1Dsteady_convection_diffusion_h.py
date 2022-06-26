# File      :1Dsteady_convection_diffusion_h.py
# Time      :2022/1/5--21:24
# Author    :JF Li
# Version   :python 3.7

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim


def analytical_solution(phi_0, phi_L, L, Pe, N, plot = False):
    def f(x):
        ret = (np.exp(Pe * x / L) - 1) / (np.exp(Pe) - 1)
        ret = ret * (phi_L - phi_0) + phi_0
        return ret

    coor = np.linspace(0, L, N + 2)
    value = np.array([f(c) for c in coor])
    if plot == True:
        plt.plot(coor, value, label = "PINN-SN", linewidth = 1, color = 'black', marker = 'o',
                 markerfacecolor = 'black', markersize = 2)
        plt.xlabel('x')
        plt.ylabel(r'$\phi$')
        plt.title('1D steady convection-diffusion eq')
        plt.legend()
        plt.savefig("./anal.png")
        plt.show()
    return coor[1:-1], value[1:-1]


class Conv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, stride, pad):
        super(Conv1d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels = in_ch, out_channels = out_ch, kernel_size = kernel, stride = stride, padding = pad),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace = True)
        )

    def forward(self, x):
        return self.conv(x)


class PICNN(nn.Module):
    def __init__(self):
        super(PICNN, self).__init__()
        self.conv1 = Conv1d(in_ch = 1, out_ch = 4, kernel = 4, stride = 1, pad = 1)
        self.conv2 = Conv1d(in_ch = 4, out_ch = 8, kernel = 4, stride = 1, pad = 1)
        self.conv3 = Conv1d(in_ch = 8, out_ch = 16, kernel = 4, stride = 1, pad = 1)
        self.upsample = nn.Upsample(size = 20)
        self.conv4 = nn.Conv1d(in_channels = 16, out_channels = 1, kernel_size = 3, stride = 1, padding = 1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.upsample(out)
        out = self.conv4(out)
        return out


class Conv1dDerivative(nn.Module):
    def __init__(self, P_delta, scheme = 'CD'):
        super(Conv1dDerivative, self).__init__()

        # 一阶迎风格式
        DerFilter_FUS = torch.tensor([[[(1 + P_delta) / (2 + P_delta), -1, 1 / (2 + P_delta)]]],
                                     dtype = torch.float32)
        self.filter_FUS = nn.Conv1d(in_channels = 1, out_channels = 1, kernel_size = 3, stride = 1, padding = 0,
                                    bias = False)
        self.filter_FUS.weight = nn.Parameter(DerFilter_FUS, requires_grad = False)

        # 二阶迎风格式
        DerFilter_SUS = torch.tensor([[[P_delta, -(4 * P_delta + 2), 3 * P_delta + 4, -2]]],
                                     dtype = torch.float32)
        self.filter_SUS = nn.Conv1d(in_channels = 1, out_channels = 1, kernel_size = 4, stride = 1, padding = 0,
                                    bias = False)
        self.filter_SUS.weight = nn.Parameter(DerFilter_SUS, requires_grad = False)

        if scheme == 'FUS':
            self.filter = self.filter_FUS
        elif scheme == 'SUS':
            self.filter = self.filter_SUS

    def forward(self, x):
        derivate = self.filter(x)
        return derivate


def phyloss(x, phi_0, phi_L, P_delta, scheme):
    left = torch.tensor([[[phi_0]]], dtype = torch.float32)
    right = torch.tensor([[[phi_L]]], dtype = torch.float32)
    x_bound = torch.cat((left, x, right), dim = 2)

    if scheme == 'SUS':
        lleft = torch.tensor([[[2 * phi_0 - x[0][0][0]]]], dtype = torch.float32)
        x_bound = torch.cat((lleft, x_bound), dim = 2)

    loss_func = Conv1dDerivative(P_delta, scheme)
    loss = loss_func(x_bound)
    mse_loss = nn.MSELoss()
    loss = mse_loss(loss, torch.zeros_like(loss))
    return loss


def frobenius_norm(tensor):
    return torch.sqrt(torch.sum(torch.pow(tensor, 2)))


def train(coor, value, model, epoch, phi_0, phi_L, P_delta, scheme, lr = 0.0001):
    optimizer = optim.Adam(model.parameters(), lr = lr)
    for i in range(epoch):
        output = model(coor)
        loss = phyloss(output, phi_0, phi_L, P_delta, scheme)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('epoch:', i, 'loss:', loss)
    # test
    pred = model(coor)
    error = frobenius_norm(pred - value) / frobenius_norm(value)
    print('predict error:', error)
    print(pred)
    return pred, error


if __name__ == '__main__':
    phi_0, phi_L, L, Pe, N = 50, 55, 2, 100, 20
    P_delta = Pe / N
    epoch = 20000
    coor, value = analytical_solution(phi_0, phi_L, L, Pe, N)
    coor = torch.tensor(coor, dtype = torch.float32).resize_((1, 1, coor.shape[0]))
    value = torch.tensor(value, dtype = torch.float32).resize_((1, 1, value.shape[0]))
    model_FUS = PICNN()
    pred_FUS, error_FUS = train(coor, value, model_FUS, epoch, phi_0, phi_L, P_delta, scheme = 'FUS')
    model_SUS = PICNN()
    pred_SUS, error_SUS = train(coor, value, model_SUS, 30000, phi_0, phi_L, P_delta, scheme = 'SUS')

    # error
    print('FUS', error_FUS, '\nSUS', error_SUS)

    # plotting
    left_x = np.array([[[0]]])
    right_x = np.array([[[L]]])
    left_v = np.array([[[phi_0]]])
    right_v = np.array([[[phi_L]]])
    coor = np.concatenate((left_x, coor.detach().numpy(), right_x), axis = 2).reshape(N + 2)
    value = np.concatenate((left_v, value.detach().numpy(), right_v), axis = 2).reshape(N + 2)
    pred_FUS = np.concatenate((left_v, pred_FUS.detach().numpy(), right_v), axis = 2).reshape(N + 2)
    pred_SUS = np.concatenate((left_v, pred_SUS.detach().numpy(), right_v), axis = 2).reshape(N + 2)
    plt.plot(coor, value, label = "analytical", linewidth = 1, color = 'black', marker = 'o',
             markerfacecolor = 'black', markersize = 2)
    plt.plot(coor, pred_FUS, label = "FUS", linewidth = 1, color = 'yellow', marker = 'o',
             markerfacecolor = 'yellow', markersize = 2)
    plt.plot(coor, pred_SUS, label = "SUS", linewidth = 1, color = 'red', marker = 'o',
             markerfacecolor = 'red', markersize = 2)
    plt.xlabel('x')
    plt.ylabel(r'$\phi$')
    plt.title('1D steady convection-diffusion equation')
    plt.legend()
    plt.savefig("./result4.png")
    plt.show()
