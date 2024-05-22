import itertools

import pytest
import torch
import torch.nn as nn

from models import fast_vggkan, vggkan, vggkaln, vggkacn, vggkagn


@pytest.mark.parametrize("dropout, dropout_linear, groups, l1_decay, vgg_type, expected_feature_shape",
                         itertools.product([0.0, 0.5], [0.0, 0.5], [1, 4], [0, 0.1],
                                           ['VGG11', 'VGG13'], [(1, 1), (7, 7)]))
def test_vggkan(dropout, dropout_linear, groups, l1_decay, vgg_type, expected_feature_shape):
    bs = 6
    spatial_dim = 64
    input_dim = 3
    num_classes = 128

    input_tensor = torch.rand((bs, input_dim, spatial_dim, spatial_dim))
    conv = vggkan(input_dim, num_classes, spline_order=3, groups=groups,
                  grid_size=5, base_activation=nn.GELU,
                  grid_range=[-1, 1], dropout=dropout, l1_decay=l1_decay,
                  vgg_type=vgg_type, expected_feature_shape=expected_feature_shape,
                  dropout_linear=dropout_linear)
    out = conv(input_tensor)
    assert out.shape == (bs, num_classes)


@pytest.mark.parametrize("dropout, dropout_linear, groups, l1_decay, vgg_type, expected_feature_shape",
                         itertools.product([0.0, 0.5], [0.0, 0.5], [1, 4], [0, 0.1],
                                           ['VGG11', 'VGG13'], [(1, 1), (7, 7)]))
def test_fastvggkan(dropout, dropout_linear, groups, l1_decay, vgg_type, expected_feature_shape):
    bs = 6
    spatial_dim = 64
    input_dim = 3
    num_classes = 128

    input_tensor = torch.rand((bs, input_dim, spatial_dim, spatial_dim))
    conv = fast_vggkan(input_dim, num_classes, groups=groups,
                       grid_size=5, base_activation=nn.GELU,
                       grid_range=[-1, 1], dropout=dropout, l1_decay=l1_decay,
                       vgg_type=vgg_type, expected_feature_shape=expected_feature_shape,
                       dropout_linear=dropout_linear)
    out = conv(input_tensor)
    assert out.shape == (bs, num_classes)


@pytest.mark.parametrize("dropout, dropout_linear, groups, l1_decay, vgg_type, expected_feature_shape",
                         itertools.product([0.0, 0.5], [0.0, 0.5], [1, 4], [0, 0.1],
                                           ['VGG11', 'VGG13'], [(1, 1), (7, 7)]))
def test_vggkaln(dropout, dropout_linear, groups, l1_decay, vgg_type, expected_feature_shape):
    bs = 6
    spatial_dim = 64
    input_dim = 3
    num_classes = 128

    input_tensor = torch.rand((bs, input_dim, spatial_dim, spatial_dim))
    conv = vggkaln(input_dim, num_classes, groups=groups, dropout=dropout, l1_decay=l1_decay,
                   vgg_type=vgg_type, expected_feature_shape=expected_feature_shape, degree=3,
                   dropout_linear=dropout_linear)
    out = conv(input_tensor)
    assert out.shape == (bs, num_classes)


@pytest.mark.parametrize("dropout, dropout_linear, groups, l1_decay, vgg_type, expected_feature_shape",
                         itertools.product([0.0, 0.5], [0.0, 0.5], [1, 4], [0, 0.1],
                                           ['VGG11', 'VGG13'], [(1, 1), (7, 7)]))
def test_vggkagn(dropout, dropout_linear, groups, l1_decay, vgg_type, expected_feature_shape):
    bs = 6
    spatial_dim = 64
    input_dim = 3
    num_classes = 128

    input_tensor = torch.rand((bs, input_dim, spatial_dim, spatial_dim))
    conv = vggkagn(input_dim, num_classes, groups=groups, dropout=dropout, l1_decay=l1_decay,
                   vgg_type=vgg_type, expected_feature_shape=expected_feature_shape, degree=3,
                   dropout_linear=dropout_linear)
    out = conv(input_tensor)
    assert out.shape == (bs, num_classes)


@pytest.mark.parametrize("dropout, dropout_linear, groups, l1_decay, vgg_type, expected_feature_shape",
                         itertools.product([0.0, 0.5], [0.0, 0.5], [1, 4], [0, 0.1],
                                           ['VGG11', 'VGG13'], [(1, 1), (7, 7)]))
def test_vggkacn(dropout, dropout_linear, groups, l1_decay, vgg_type, expected_feature_shape):
    bs = 6
    spatial_dim = 64
    input_dim = 3
    num_classes = 128

    input_tensor = torch.rand((bs, input_dim, spatial_dim, spatial_dim))
    conv = vggkacn(input_dim, num_classes, groups=groups, dropout=dropout, l1_decay=l1_decay,
                   vgg_type=vgg_type, expected_feature_shape=expected_feature_shape, degree=3,
                   dropout_linear=dropout_linear)
    out = conv(input_tensor)
    assert out.shape == (bs, num_classes)