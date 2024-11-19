import torch
from torch import Tensor
import torch.nn as nn
from typing import List, Tuple, Union, Optional, Callable, Type


class KANConvNDLayer(nn.Module):
    def __init__(
            self,
            conv_class: Type[nn.Module],
            norm_class: Type[nn.Module],
            ndim: int,
            in_channels: int,
            out_channels: int,
            spline_order: int,
            kernel_size: Union[int, Tuple[int, ...]],
            stride: Union[int, Tuple[int, ...]],
            padding: Union[int, Tuple[int, ...]],
            dilation: Union[int, Tuple[int, ...]],
            groups: int = 1,
            grid_size: int = 5,
            base_activation: Callable[[Tensor], Tensor] = nn.GELU(),
            grid_range: Optional[List[float]] = None,
            dropout: float = 0.,
            **norm_kwargs
    ):
        super(KANConvNDLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spline_order = spline_order
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.dilation = dilation
        self.groups = groups
        self.ndim = ndim
        self.grid_size = grid_size
        self.base_activation = base_activation
        if grid_range is None:
            grid_range = [-1, 1]
        self.grid_range = grid_range
        self.norm_kwargs = norm_kwargs
        self.dropout = None
        if dropout > 0:
            if ndim == 1:
                self.dropout = nn.Dropout1d(p=dropout)
            if ndim == 2:
                self.dropout = nn.Dropout2d(p=dropout)
            if ndim == 3:
                self.dropout = nn.Dropout3d(p=dropout)
        if groups <= 0:
            raise ValueError('groups must be a positive integer')
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')

        self.base_conv = nn.ModuleList([
                                           conv_class(
                                               in_channels // groups,
                                               out_channels // groups,
                                               kernel_size,
                                               stride,
                                               padding,
                                               dilation,
                                               groups=1,
                                               bias=False,
                                           )
                                       ] * groups)

        self.spline_conv = nn.ModuleList([
                                             conv_class(
                                                 (grid_size + spline_order) * in_channels // groups,
                                                 out_channels // groups,
                                                 kernel_size,
                                                 stride,
                                                 padding,
                                                 dilation,
                                                 groups=1,
                                                 bias=False
                                             )
                                         ] * groups)

        self.layer_norm = nn.ModuleList([
                                            norm_class(out_channels // groups, **norm_kwargs)
                                        ] * groups)

        self.prelus = nn.ModuleList([nn.PReLU()] * groups)

        h = (self.grid_range[1] - self.grid_range[0]) / grid_size
        self.grid = torch.linspace(
            self.grid_range[0] - h * spline_order,
            self.grid_range[1] + h * spline_order,
            grid_size + 2 * spline_order + 1,
            dtype=torch.float32
        )
        # Initialize weights using Kaiming uniform distribution for better training start
        for conv_layer in self.base_conv:
            nn.init.kaiming_uniform_(conv_layer.weight, nonlinearity='linear')

        for conv_layer in self.spline_conv:
            nn.init.kaiming_uniform_(conv_layer.weight, nonlinearity='linear')

    def forward_kan(self, x: Tensor, group_index: int) -> Tensor:

        # Apply base activation to input and then linear transform with base weights
        base_output = self.base_conv[group_index](self.base_activation(x))

        x_uns = x.unsqueeze(-1)  # Expand dimensions for spline operations.
        # Compute the basis for the spline using intervals and input values.
        target = x.shape[1:] + self.grid.shape
        grid = self.grid.view([1] * (self.ndim + 1) + [-1])
        grid = grid.expand(target).contiguous().to(x.device)

        bases = ((x_uns >= grid[..., :-1]) & (x_uns < grid[..., 1:])).to(x.dtype)

        # Compute the spline basis over multiple orders.
        for k in range(1, self.spline_order + 1):
            left_intervals = grid[..., :-(k + 1)]
            right_intervals = grid[..., k:-1]
            delta = torch.where(right_intervals == left_intervals, torch.ones_like(right_intervals),
                                right_intervals - left_intervals)
            bases = ((x_uns - left_intervals) / delta * bases[..., :-1]) + \
                    ((grid[..., k + 1:] - x_uns) / (grid[..., k + 1:] - grid[..., 1:(-k)]) * bases[..., 1:])
        bases = bases.contiguous()
        bases = bases.moveaxis(-1, 2).flatten(1, 2)
        spline_output = self.spline_conv[group_index](bases)
        x = self.prelus[group_index](self.layer_norm[group_index](base_output + spline_output))

        if self.dropout:
            x = self.dropout(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        split_x = torch.split(x, self.in_channels // self.groups, dim=1)
        output = []
        for group_ind, xy in enumerate(split_x):
            y = self.forward_kan(xy, group_ind)
            output.append(y.clone())
        y = torch.cat(output, dim=1)
        return y


class KANConv3DLayer(KANConvNDLayer):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[int, Tuple[int, int, int]],
            stride: Union[int, Tuple[int, int, int]] = 1,
            padding: Union[int, Tuple[int, int, int]] = 0,
            dilation: Union[int, Tuple[int, int, int]] = 1,
            groups: int = 1,
            spline_order: int = 3,
            grid_size: int = 5,
            base_activation: Callable[[Tensor], Tensor] = nn.GELU(),
            grid_range: Optional[List[float]] = None,
            dropout: float = 0.,
            norm_class: Type[nn.Module] = nn.InstanceNorm3d,
            **norm_kwargs
    ):
        super(KANConv3DLayer, self).__init__(
            conv_class=nn.Conv3d,
            norm_class=norm_class,
            ndim=3,
            in_channels=in_channels,
            out_channels=out_channels,
            spline_order=spline_order,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            grid_size=grid_size,
            base_activation=base_activation,
            grid_range=grid_range,
            dropout=dropout,
            **norm_kwargs
        )


class KANConv2DLayer(KANConvNDLayer):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[int, Tuple[int, int]],
            stride: Union[int, Tuple[int, int]] = 1,
            padding: Union[int, Tuple[int, int]] = 0,
            dilation: Union[int, Tuple[int, int]] = 1,
            groups: int = 1,
            spline_order: int = 3,
            grid_size: int = 5,
            base_activation: Callable[[Tensor], Tensor] = nn.GELU(),
            grid_range: Optional[List[float]] = None,
            dropout: float = 0.,
            norm_class: Type[nn.Module] = nn.InstanceNorm2d,
            **norm_kwargs
    ):
        super(KANConv2DLayer, self).__init__(
            conv_class=nn.Conv2d,
            norm_class=norm_class,
            ndim=2,
            in_channels=in_channels,
            out_channels=out_channels,
            spline_order=spline_order,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            grid_size=grid_size,
            base_activation=base_activation,
            grid_range=grid_range,
            dropout=dropout,
            **norm_kwargs
        )


class KANConv1DLayer(KANConvNDLayer):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: int = 0,
            dilation: int = 1,
            groups: int = 1,
            spline_order: int = 3,
            grid_size: int = 5,
            base_activation: Callable[[Tensor], Tensor] = nn.GELU(),
            grid_range: Optional[List[float]] = None,
            dropout: float = 0.,
            norm_class: Type[nn.Module] = nn.InstanceNorm1d,
            **norm_kwargs
    ):
        super(KANConv1DLayer, self).__init__(
            conv_class=nn.Conv1d,
            norm_class=norm_class,
            ndim=1,
            in_channels=in_channels,
            out_channels=out_channels,
            spline_order=spline_order,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            grid_size=grid_size,
            base_activation=base_activation,
            grid_range=grid_range,
            dropout=dropout,
            **norm_kwargs
        )
