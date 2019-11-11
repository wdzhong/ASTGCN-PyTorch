# -*- coding:utf-8 -*-

import sys
import numpy as np

sys.path.append('.')


def test_ASTGCN_submodule():
    from model.astgcn import ASTGCN_submodule
    import torch
    x = torch.randn(32, 307, 3, 24)
    K = 3
    cheb_polynomials = [torch.randn(307, 307) for i in range(K)]
    backbone = [
        {
            "K": K,
            "num_of_chev_filters": 64,
            "num_of_time_filters": 64,
            "time_conv_strides": 2,
            "cheb_polynomials": cheb_polynomials
        },
        {
            "K": K,
            "num_of_chev_filters": 64,
            "num_of_time_filters": 64,
            "time_conv_strides": 1,
            "cheb_polynomials": cheb_polynomials
        }
    ]
    net = ASTGCN_submodule(12, backbone, 307, 3, [24, 12])

    output = net(x)
    assert output.shape == (32, 307, 12)
    assert type(output.detach().numpy().mean()) == np.float32


def test_predict1():
    from model.astgcn import ASTGCN
    from model.model_config import get_backbones
    import torch
    device = torch.device('cpu')
    all_backbones = get_backbones('configurations/PEMS04.conf',
                                  'data/PEMS04/distance.csv', device)

    net = ASTGCN(12, all_backbones, 307, 3, [[24, 12], [12, 12], [24, 12]], device)

    test_w = torch.randn(16, 307, 3, 24).to(device)
    test_d = torch.randn(16, 307, 3, 12).to(device)
    test_r = torch.randn(16, 307, 3, 24).to(device)
    output = net([test_w, test_d, test_r])
    assert output.shape == (16, 307, 12)
    assert type(output.detach().numpy().mean()) == np.float32


def test_predict2():
    from model.astgcn import ASTGCN
    from model.model_config import get_backbones
    import torch
    device = torch.device('cpu')
    all_backbones = get_backbones('configurations/PEMS08.conf',
                                  'data/PEMS08/distance.csv', device)

    net = ASTGCN(12, all_backbones, 170, 3, [[12, 12], [12, 12], [36, 12]], device)

    test_w = torch.randn(8, 170, 3, 12).to(device)
    test_d = torch.randn(8, 170, 3, 12).to(device)
    test_r = torch.randn(8, 170, 3, 36).to(device)
    output = net([test_w, test_d, test_r])
    assert output.shape == (8, 170, 12)
    assert type(output.detach().numpy().mean()) == np.float32
