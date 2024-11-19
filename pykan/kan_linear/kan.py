# taken from and based on https://github.com/1ssb/torchkan/blob/main/torchkan.py
# and https://github.com/1ssb/torchkan/blob/main/KALnet.py
# and https://github.com/ZiyaoLi/fast-kan/blob/master/fastkan/fastkan.py
# Copyright 2024 Li, ZiYao
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# and https://github.com/SynodicMonth/ChebyKAN/blob/main/ChebyKANLayer.py
# and https://github.com/Khochawongwat/GRAMKAN/blob/main/model.py
# and https://github.com/zavareh1/Wav-KAN
# and https://github.com/quiqi/relu_kan/issues/2

import itertools

from torch import Tensor
import torch.nn as nn
from typing import List, Callable, Optional

from .kan_layers import (
    KANLayer,
    KALNLayer,
    FastKANLayer,
    ChebyKANLayer,
    GRAMLayer,
    WavKANLayer,
    JacobiKANLayer,
    BernsteinKANLayer,
    ReLUKANLayer,
    BottleNeckGRAMLayer,
)
from ..utils import L1


class KAN(nn.Module):  # Kolmogorov Arnold Legendre Network (KAL-Net)
    def __init__(
            self,
            hidden_layers: List[int],
            dropout: float = 0.,
            grid_size: int = 5,
            spline_order: int = 3,
            base_activation: Callable[[Tensor], Tensor] = nn.GELU(),
            grid_range: Optional[List[float]] = None,
            l1_decay: float = 0.,
            first_dropout: bool = True
    ):
        super(KAN, self).__init__()  # Initialize the parent nn.Module class

        # hidden_layers: A list of integers specifying the number of neurons in each layer
        self.hidden_layers = hidden_layers
        # polynomial_order: Order up to which Legendre polynomials are calculated
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.base_activation = base_activation
        if grid_range is None:
            grid_range = [-1, 1]
        self.grid_range = grid_range

        # list of layers
        self.layers = nn.ModuleList()
        if dropout > 0 and first_dropout:
            self.layers.append(nn.Dropout(p=dropout))
        self.base_activation = base_activation
        self.num_layers = len(hidden_layers[:-1])

        for i, (in_features, out_features) in enumerate(itertools.pairwise(hidden_layers)):
            layer = KANLayer(
                in_features,
                out_features,
                grid_size=grid_size,
                spline_order=spline_order,
                base_activation=base_activation,
                grid_range=grid_range
            )
            if l1_decay > 0 and i != self.num_layers - 1:
                layer = L1(layer, l1_decay)
            self.layers.append(layer)
            if dropout > 0 and i != self.num_layers - 1:
                self.layers.append(nn.Dropout(p=dropout))

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class KALN(nn.Module):  # Kolmogorov Arnold Legendre Network (KAL-Net)
    def __init__(
            self,
            hidden_layers: List[int],
            dropout: float = 0.,
            l1_decay: float = 0.,
            degree: int = 3,
            base_activation: Callable[[Tensor], Tensor] = nn.SiLU(),
            first_dropout: bool = True
    ):
        super(KALN, self).__init__()  # Initialize the parent nn.Module class

        # hidden_layers: A list of integers specifying the number of neurons in each layer
        self.hidden_layers = hidden_layers
        # polynomial_order: Order up to which Legendre polynomials are calculated
        self.polynomial_order = degree
        # list of layers
        self.layers = nn.ModuleList()
        if dropout > 0 and first_dropout:
            self.layers.append(nn.Dropout(p=dropout))
        self.base_activation = base_activation
        self.num_layers = len(hidden_layers[:-1])

        for i, (in_features, out_features) in enumerate(itertools.pairwise(hidden_layers)):
            # Base weight for linear transformation in each layer
            layer = KALNLayer(
                in_features,
                out_features,
                degree=degree,
                base_activation=base_activation
            )
            if l1_decay > 0 and i != self.num_layers - 1:
                layer = L1(layer, l1_decay)
            self.layers.append(layer)
            if dropout > 0 and i != self.num_layers - 1:
                self.layers.append(nn.Dropout(p=dropout))

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class FastKAN(nn.Module):
    def __init__(
            self,
            hidden_layers: List[int],
            dropout: float = 0.,
            l1_decay: float = 0.,
            grid_range: Optional[List[float]] = None,
            grid_size: int = 8,
            use_base_update: bool = True,
            base_activation: Callable[[Tensor], Tensor] = nn.SiLU(),
            spline_weight_init_scale: float = 0.1,
            first_dropout: bool = True
    ):
        super().__init__()
        self.hidden_layers = hidden_layers
        if grid_range is None:
            grid_range = [-2, 2]
        self.grid_min = grid_range[0]
        self.grid_max = grid_range[1]
        self.use_base_update = use_base_update
        self.base_activation = base_activation
        self.spline_weight_init_scale = spline_weight_init_scale
        self.num_layers = len(hidden_layers[:-1])

        self.layers = nn.ModuleList()
        if dropout > 0 and first_dropout:
            self.layers.append(nn.Dropout(p=dropout))

        for i, (in_features, out_features) in enumerate(itertools.pairwise(hidden_layers)):
            # Base weight for linear transformation in each layer
            layer = FastKANLayer(
                in_features,
                out_features,
                grid_min=self.grid_min,
                grid_max=self.grid_max,
                num_grids=grid_size,
                use_base_update=use_base_update,
                base_activation=base_activation,
                spline_weight_init_scale=spline_weight_init_scale
            )
            if l1_decay > 0 and i != self.num_layers - 1:
                layer = L1(layer, l1_decay)
            self.layers.append(layer)

            if dropout > 0 and i != self.num_layers - 1:
                self.layers.append(nn.Dropout(p=dropout))

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class KACN(nn.Module):  # Kolmogorov Arnold Legendre Network (KAL-Net)
    def __init__(
            self,
            hidden_layers: List[int],
            dropout: float = 0.,
            degree: int = 3,
            l1_decay: float = 0.,
            first_dropout: bool = True
    ):
        super(KACN, self).__init__()  # Initialize the parent nn.Module class

        # hidden_layers: A list of integers specifying the number of neurons in each layer
        self.hidden_layers = hidden_layers
        # polynomial_order: Order up to which Legendre polynomials are calculated
        self.polynomial_order = degree
        # list of layers
        self.layers = nn.ModuleList()
        if dropout > 0 and first_dropout:
            self.layers.append(nn.Dropout(p=dropout))
        self.num_layers = len(hidden_layers[:-1])

        for i, (in_features, out_features) in enumerate(itertools.pairwise(hidden_layers)):
            # Base weight for linear transformation in each layer
            layer = ChebyKANLayer(in_features, out_features, degree=degree)
            if l1_decay > 0 and i != self.num_layers - 1:
                layer = L1(layer, l1_decay)
            self.layers.append(layer)

            if dropout > 0 and i != self.num_layers - 1:
                self.layers.append(nn.Dropout(p=dropout))

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class KAGN(nn.Module):
    def __init__(
            self,
            hidden_layers: List[int],
            dropout: float = 0.,
            degree: int = 3,
            base_activation: Callable[[Tensor], Tensor] = nn.SiLU(),
            l1_decay: float = 0.,
            first_dropout: bool = True
    ):
        super(KAGN, self).__init__()  # Initialize the parent nn.Module class

        # hidden_layers: A list of integers specifying the number of neurons in each layer
        self.hidden_layers = hidden_layers
        # polynomial_order: Order up to which Legendre polynomials are calculated
        self.polynomial_order = degree
        # list of layers
        self.layers = nn.ModuleList()
        if dropout > 0 and first_dropout:
            self.layers.append(nn.Dropout(p=dropout))
        self.base_activation = base_activation
        self.num_layers = len(hidden_layers[:-1])

        for i, (in_features, out_features) in enumerate(itertools.pairwise(hidden_layers)):
            # Base weight for linear transformation in each layer
            layer = GRAMLayer(
                in_features,
                out_features,
                degree=degree,
                base_activation=base_activation
            )
            if l1_decay > 0 and i != self.num_layers - 1:
                layer = L1(layer, l1_decay)
            self.layers.append(layer)

            if dropout > 0 and i != self.num_layers - 1:
                self.layers.append(nn.Dropout(p=dropout))

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class BottleNeckKAGN(nn.Module):
    def __init__(
            self,
            hidden_layers: List[int],
            dropout: float = 0.,
            degree: int = 3,
            base_activation: Callable[[Tensor], Tensor] = nn.SiLU(),
            l1_decay: float = 0.,
            first_dropout: bool = True,
            dim_reduction: float = 8,
            min_internal: int = 16
    ):
        super(BottleNeckKAGN, self).__init__()  # Initialize the parent nn.Module class

        # hidden_layers: A list of integers specifying the number of neurons in each layer
        self.hidden_layers = hidden_layers
        # polynomial_order: Order up to which Legendre polynomials are calculated
        self.polynomial_order = degree
        # list of layers
        self.layers = nn.ModuleList()
        if dropout > 0 and first_dropout:
            self.layers.append(nn.Dropout(p=dropout))
        self.base_activation = base_activation
        self.num_layers = len(hidden_layers[:-1])

        for i, (in_features, out_features) in enumerate(itertools.pairwise(hidden_layers)):
            # Base weight for linear transformation in each layer
            layer = BottleNeckGRAMLayer(
                in_features,
                out_features,
                degree=degree,
                base_activation=base_activation,
                dim_reduction=dim_reduction,
                min_internal=min_internal
            )
            if l1_decay > 0 and i != self.num_layers - 1:
                layer = L1(layer, l1_decay)
            self.layers.append(layer)

            if dropout > 0 and i != self.num_layers - 1:
                self.layers.append(nn.Dropout(p=dropout))

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class KABN(nn.Module):
    def __init__(
            self,
            hidden_layers: List[int],
            dropout: float = 0.,
            degree: int = 3,
            base_activation: Callable[[Tensor], Tensor] = nn.SiLU(),
            l1_decay: float = 0.,
            first_dropout: bool = True
    ):
        super(KABN, self).__init__()  # Initialize the parent nn.Module class

        # hidden_layers: A list of integers specifying the number of neurons in each layer
        self.hidden_layers = hidden_layers
        # polynomial_order: Order up to which Legendre polynomials are calculated
        self.polynomial_order = degree
        # list of layers
        self.layers = nn.ModuleList()
        if dropout > 0 and first_dropout:
            self.layers.append(nn.Dropout(p=dropout))
        self.base_activation = base_activation
        self.num_layers = len(hidden_layers[:-1])

        for i, (in_features, out_features) in enumerate(itertools.pairwise(hidden_layers)):
            # Base weight for linear transformation in each layer
            layer = BernsteinKANLayer(
                in_features,
                out_features,
                degree=degree,
                base_activation=base_activation
            )
            if l1_decay > 0 and i != self.num_layers - 1:
                layer = L1(layer, l1_decay)
            self.layers.append(layer)

            if dropout > 0 and i != self.num_layers - 1:
                self.layers.append(nn.Dropout(p=dropout))

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class KAJN(nn.Module):
    def __init__(
            self,
            hidden_layers: List[int],
            dropout: float = 0.,
            l1_decay: float = 0.,
            degree: int = 3,
            a: float = 1,
            b: float = 1,
            base_activation: Callable[[Tensor], Tensor] = nn.SiLU(),
            first_dropout: bool = True
    ):
        super(KAJN, self).__init__()  # Initialize the parent nn.Module class

        # hidden_layers: A list of integers specifying the number of neurons in each layer
        self.hidden_layers = hidden_layers
        # polynomial_order: Order up to which Legendre polynomials are calculated
        self.polynomial_order = degree
        # list of layers
        self.layers = nn.ModuleList()
        if dropout > 0 and first_dropout:
            self.layers.append(nn.Dropout(p=dropout))
        self.base_activation = base_activation
        self.num_layers = len(hidden_layers[:-1])

        for i, (in_features, out_features) in enumerate(itertools.pairwise(hidden_layers)):
            # Base weight for linear transformation in each layer
            layer = JacobiKANLayer(
                in_features,
                out_features,
                degree=degree,
                a=a, b=b,
                base_activation=base_activation
            )
            if l1_decay > 0 and i != self.num_layers - 1:
                layer = L1(layer, l1_decay)
            self.layers.append(layer)

            if dropout > 0 and i != self.num_layers - 1:
                self.layers.append(nn.Dropout(p=dropout))

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class WavKAN(nn.Module):
    def __init__(
            self,
            hidden_layers: List[int],
            dropout: float = 0.,
            l1_decay: float = 0.,
            first_dropout: bool = True,
            wavelet_type: str = 'mexican_hat'
    ):
        super(WavKAN, self).__init__()  # Initialize the parent nn.Module class

        if wavelet_type not in ['mexican_hat', 'morlet', 'dog', 'meyer', 'shannon']:
            raise TypeError(f'Unsupported wavelet type: {wavelet_type}')
        # hidden_layers: A list of integers specifying the number of neurons in each layer
        self.hidden_layers = hidden_layers
        # polynomial_order: Order up to which Legendre polynomials are calculated
        self.wavelet_type = wavelet_type
        # list of layers
        self.layers = nn.ModuleList()
        if dropout > 0 and first_dropout:
            self.layers.append(nn.Dropout(p=dropout))
        self.num_layers = len(hidden_layers[:-1])

        for i, (in_features, out_features) in enumerate(itertools.pairwise(hidden_layers)):
            # Base weight for linear transformation in each layer
            layer = WavKANLayer(in_features, out_features, wavelet_type=wavelet_type)
            if l1_decay > 0 and i != self.num_layers - 1:
                layer = L1(layer, l1_decay)
            self.layers.append(layer)

            if dropout > 0 and i != self.num_layers - 1:
                self.layers.append(nn.Dropout(p=dropout))

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class ReLUKAN(nn.Module):
    def __init__(
            self,
            hidden_layers: List[int],
            dropout: float = 0.0,
            l1_decay: float = 0.0,
            g: int = 1,
            k: int = 1,
            train_ab: bool = True,
            first_dropout: bool = True
    ):
        super(ReLUKAN, self).__init__()  # Initialize the parent nn.Module class

        # hidden_layers: A list of integers specifying the number of neurons in each layer
        self.hidden_layers = hidden_layers
        # list of layers
        self.layers = nn.ModuleList()
        if dropout > 0 and first_dropout:
            self.layers.append(nn.Dropout(p=dropout))
        self.num_layers = len(hidden_layers[:-1])

        for i, (in_features, out_features) in enumerate(itertools.pairwise(hidden_layers)):
            # Base weight for linear transformation in each layer
            layer = ReLUKANLayer(
                in_features,
                out_features,
                g=g, k=k,
                train_ab=train_ab
            )
            if l1_decay > 0 and i != self.num_layers - 1:
                layer = L1(layer, l1_decay)
            self.layers.append(layer)

            if dropout > 0 and i != self.num_layers - 1:
                self.layers.append(nn.Dropout(p=dropout))

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
