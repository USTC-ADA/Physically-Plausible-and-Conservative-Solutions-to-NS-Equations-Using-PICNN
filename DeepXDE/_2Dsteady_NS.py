# File      :_2Dsteady_NS.py
# Time      :2022/11/9--16:49
# Author    :JF Li
# Version   :python 3.7

import deepxde as dde
import set_bc as bc
import test

rho = 1
data_size = 51
mu = 0.001
data_path = './data/data8.txt'


def pde(coor, value):
    '''NS equation structure'''
    u, v, p = value[:, 0:1], value[:, 1:2], value[:, 2:3]
    u_x = dde.grad.jacobian(value, coor, i = 0, j = 0)
    u_y = dde.grad.jacobian(value, coor, i = 0, j = 1)
    u_xx = dde.grad.hessian(value, coor, component = 0, i = 0, j = 0)
    u_yy = dde.grad.hessian(value, coor, component = 0, i = 1, j = 1)

    v_x = dde.grad.jacobian(value, coor, i = 1, j = 0)
    v_y = dde.grad.jacobian(value, coor, i = 1, j = 1)
    v_xx = dde.grad.hessian(value, coor, component = 1, i = 0, j = 0)
    v_yy = dde.grad.hessian(value, coor, component = 1, i = 1, j = 1)

    p_x = dde.grad.jacobian(value, coor, i = 2, j = 0)
    p_y = dde.grad.jacobian(value, coor, i = 2, j = 1)

    f = rho * (u * u_x + v * u_y) + p_x - mu * (u_xx + u_yy)
    g = rho * (u * v_x + v * v_y) + p_y - mu * (v_xx + v_yy)
    mass = rho * (u_x + v_y)
    return [f, g, mass]


def main():
    # load test dataset
    coor, value = test.data_extraction(data_path, data_size)
    coor, value = test.getDataset(coor, value, data_size)
    # set problem
    domain = dde.geometry.Rectangle(xmin = [0, 0], xmax = [1, 1])
    boundary_u_zero = dde.DirichletBC(domain, bc.func_u_zero, bc.pos_u_zero, component = 0)
    boundary_u = dde.DirichletBC(domain, bc.func_u, bc.pos_u, component = 0)
    boundary_v_zero = dde.DirichletBC(domain, bc.func_v_zero, bc.pos_v_zero, component = 1)
    boundary_v = dde.NeumannBC(domain, bc.func_v, bc.pos_v, component = 1)
    boundary_p = dde.DirichletBC(domain, bc.func_p, bc.pos_p, component = 2)
    data = dde.data.PDE(domain, pde, [boundary_u_zero, boundary_u, boundary_v_zero, boundary_v, boundary_p],
                        num_domain = 30000, num_boundary = 3000)  # 2401 200
    net = dde.maps.FNN([2] + [50] * 4 + [3], "tanh", "Glorot normal")
    # set model and train
    model = dde.Model(data, net)
    model.compile("adam", lr = 1e-3)
    model.train(epochs = 30000)
    model.compile("L-BFGS")
    model.train()
    # model.save('./model/deepxde.pt')
    # loss_history, train_state = model.train()
    # dde.saveplot(loss_history, train_state, issave = True, isplot = True)
    # predict
    # model.restore('./model/')
    value_pred = model.predict(coor)
    error_u, error_v, error_uv, error_p = test.test(value, value_pred)
    print('predict error of u', error_u)
    print('predict error of v', error_v)
    print('predict error of uv', error_uv)
    print('predict error of p', error_p)


if __name__ == '__main__':
    main()
