'''
Author: Pan Shi <shipan76@mail.ustc.edu.cn>
Date: 2023-12-18 16:42:44
LastEditTime: 2023-12-18 23:36:13
LastEditors: Pan Shi
Description: 
FilePath: /shipan/bevfusion-main/mmdet3d/models/com/V2VNet.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Sequence
import math

import collections
from itertools import repeat
from functools import partial

import torch
import torch.nn.functional as F

try:
    # pytorch<=0.4.1
    from torch.nn._functions.thnn import rnnFusedPointwise as fusedBackend
except ImportError:
    fusedBackend = None
from torch.nn import Parameter
from torch.nn.utils.rnn import PackedSequence
""" Copied from torch.nn.modules.utils """


from .Backbone import (
    LidarEncoder,
    Conv2DBatchNormRelu,
    Sparsemax,
)


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)


class V2VNet(nn.Module):
    """V2V Net

    https://arxiv.org/abs/2008.07519

    """

    def __init__(
        self,
        config,
        gnn_iter_times=3,
        in_channels=13,
        num_agent=5,
        compress_level=0,
        only_v2i=False,
    ):
        super().__init__()

        self.layer_channel = config['in_channels']
        self.gnn_iter_num = gnn_iter_times
        self.convgru = Conv2dGRU(
            in_channels=self.layer_channel * 2,
            out_channels=self.layer_channel,
            kernel_size=3,
            num_layers=1,
            bidirectional=False,
            dilation=1,
            stride=1,
        )
        self.compress_level = compress_level
        
        self.downsample = Conv2DBatchNormRelu(self.layer_channel, self.layer_channel, k_size=3, stride=2, padding=1)
        # self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
        self.downsample_2 = Conv2DBatchNormRelu(self.layer_channel, self.layer_channel, k_size=3, stride=2, padding=1)
        # self.upsample_2 = nn.Upsample(scale_factor=2, mode='nearest')
        
        self.downsample_3 = Conv2DBatchNormRelu(self.layer_channel, self.layer_channel, k_size=3, stride=2, padding=1)
        # self.upsample_3 = nn.Upsample(scale_factor=2, mode='nearest')
        
        self.downsample_4 = Conv2DBatchNormRelu(self.layer_channel, self.layer_channel, k_size=3, stride=2, padding=1)
        # self.upsample_4 = nn.Upsample(scale_factor=2, mode='nearest')
        
        self.downsample_5 = Conv2DBatchNormRelu(self.layer_channel, self.layer_channel, k_size=3, stride=2, padding=1)
        # self.upsample_5 = nn.Upsample(scale_factor=2, mode='nearest')
        
        
        self.upsample = nn.Sequential(
                nn.Upsample(
                    scale_factor=2,
                    mode="bilinear",
                    align_corners=True,
                ),
                nn.Conv2d(self.layer_channel, self.layer_channel, 3, padding=1, bias=False),
                nn.BatchNorm2d(self.layer_channel),
                nn.ReLU(True),
            )

        self.upsample_2 = nn.Sequential(
                nn.Upsample(
                    scale_factor=2,
                    mode="bilinear",
                    align_corners=True,
                ),
                nn.Conv2d(self.layer_channel, self.layer_channel, 3, padding=1, bias=False),
                nn.BatchNorm2d(self.layer_channel),
                nn.ReLU(True),
            )
        self.upsample_3 = nn.Sequential(
                nn.Upsample(
                    scale_factor=2,
                    mode="bilinear",
                    align_corners=True,
                ),
                nn.Conv2d(self.layer_channel, self.layer_channel, 3, padding=1, bias=False),
                nn.BatchNorm2d(self.layer_channel),
                nn.ReLU(True),
            )
        self.upsample_4 = nn.Sequential(
                nn.Upsample(
                    scale_factor=2,
                    mode="bilinear",
                    align_corners=True,
                ),
                nn.Conv2d(self.layer_channel, self.layer_channel, 3, padding=1, bias=False),
                nn.BatchNorm2d(self.layer_channel),
                nn.ReLU(True),
            )
        
        self.upsample_5 = nn.Sequential(
                nn.Upsample(
                    scale_factor=2,
                    mode="bilinear",
                    align_corners=True,
                ),
                nn.Conv2d(self.layer_channel, self.layer_channel, 3, padding=1, bias=False),
                nn.BatchNorm2d(self.layer_channel),
                nn.ReLU(True),
            )
        
        
        

    def forward(self, x, training=True):

        B, N, C, H, W = x.size()
        
        x = x.view(B * N, C, H, W)
        x = self.downsample(x)
        x = self.downsample_2(x)
        x = self.downsample_3(x)
        # x = self.downsample_4(x)
        # x = self.downsample_5(x)
        
        _, C, H, W = x.size()
        x = x.view(B, N, C, H, W)
        
        
        # local_com_mat_update = x
        local_com_mat_update = x.clone()

        for b in range(B):   
            agent_feat_list = list()
            for nb in range(N):
                agent_feat_list.append(x[b, nb])
                
            for _ in range(self.gnn_iter_num):

                updated_feats_list = []

                for i in range(N):
                    self.neighbor_feat_list = []
                    for j in range(N):
                        if j != i:
                            warp_feat = x[b, j]
                            self.neighbor_feat_list.append(warp_feat)

                    mean_feat = torch.mean(
                        torch.stack(self.neighbor_feat_list), dim=0
                    )  # [c, h, w]
                    cat_feat = torch.cat([agent_feat_list[i], mean_feat], dim=0)
                    cat_feat = cat_feat.unsqueeze(0).unsqueeze(0)  # [1, 1, c, h, w]
                    updated_feat, _ = self.convgru(cat_feat, None)
                    updated_feat = torch.squeeze(
                        torch.squeeze(updated_feat, 0), 0
                    )  # [c, h, w]
                    updated_feats_list.append(updated_feat)

                agent_feat_list = updated_feats_list

            for k in range(N):
                local_com_mat_update[b, k] = agent_feat_list[k]
        
        B, N, C, H, W = local_com_mat_update.size()
        local_com_mat_update = local_com_mat_update.view(B * N, C, H, W)
        local_com_mat_update = self.upsample(local_com_mat_update)
        local_com_mat_update = self.upsample_2(local_com_mat_update)
        local_com_mat_update = self.upsample_3(local_com_mat_update)
        # local_com_mat_update = self.upsample_4(local_com_mat_update)
        # local_com_mat_update = self.upsample_5(local_com_mat_update)
        
        _, C, H, W = local_com_mat_update.size()
        local_com_mat_update = local_com_mat_update.view(B, N, C, H, W)
        
        return local_com_mat_update
        # return torch.mean(local_com_mat_update, dim=1)

class ConvNdRNNBase(nn.Module):
    def __init__(
        self,
        mode: str,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Sequence[int]],
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: bool = False,
        convndim: int = 2,
        stride: Union[int, Sequence[int]] = 1,
        dilation: Union[int, Sequence[int]] = 1,
        groups: int = 1,
    ):
        super().__init__()
        self.mode = mode
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.convndim = convndim

        if convndim == 1:
            ntuple = _single
        elif convndim == 2:
            ntuple = _pair
        elif convndim == 3:
            ntuple = _triple
        else:
            raise ValueError("convndim must be 1, 2, or 3, but got {}".format(convndim))

        self.kernel_size = ntuple(kernel_size)
        self.stride = ntuple(stride)
        self.dilation = ntuple(dilation)

        self.groups = groups

        num_directions = 2 if bidirectional else 1

        if mode in ("LSTM", "PeepholeLSTM"):
            gate_size = 4 * out_channels
        elif mode == "GRU":
            gate_size = 3 * out_channels
        else:
            gate_size = out_channels

        self._all_weights = []
        for layer in range(num_layers):
            for direction in range(num_directions):
                layer_input_size = (
                    in_channels if layer == 0 else out_channels * num_directions
                )
                w_ih = Parameter(
                    torch.Tensor(
                        gate_size, layer_input_size // groups, *self.kernel_size
                    )
                )
                w_hh = Parameter(
                    torch.Tensor(gate_size, out_channels // groups, *self.kernel_size)
                )

                b_ih = Parameter(torch.Tensor(gate_size))
                b_hh = Parameter(torch.Tensor(gate_size))

                if mode == "PeepholeLSTM":
                    w_pi = Parameter(
                        torch.Tensor(
                            out_channels, out_channels // groups, *self.kernel_size
                        )
                    )
                    w_pf = Parameter(
                        torch.Tensor(
                            out_channels, out_channels // groups, *self.kernel_size
                        )
                    )
                    w_po = Parameter(
                        torch.Tensor(
                            out_channels, out_channels // groups, *self.kernel_size
                        )
                    )
                    layer_params = (w_ih, w_hh, w_pi, w_pf, w_po, b_ih, b_hh)
                    param_names = [
                        "weight_ih_l{}{}",
                        "weight_hh_l{}{}",
                        "weight_pi_l{}{}",
                        "weight_pf_l{}{}",
                        "weight_po_l{}{}",
                    ]
                else:
                    layer_params = (w_ih, w_hh, b_ih, b_hh)
                    param_names = ["weight_ih_l{}{}", "weight_hh_l{}{}"]
                if bias:
                    param_names += ["bias_ih_l{}{}", "bias_hh_l{}{}"]

                suffix = "_reverse" if direction == 1 else ""
                param_names = [x.format(layer, suffix) for x in param_names]

                for name, param in zip(param_names, layer_params):
                    setattr(self, name, param)
                self._all_weights.append(param_names)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.out_channels)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def check_forward_args(self, input, hidden, batch_sizes):
        is_input_packed = batch_sizes is not None
        expected_input_dim = (2 if is_input_packed else 3) + self.convndim
        if input.dim() != expected_input_dim:
            raise RuntimeError(
                "input must have {} dimensions, got {}".format(
                    expected_input_dim, input.dim()
                )
            )
        ch_dim = 1 if is_input_packed else 2
        if self.in_channels != input.size(ch_dim):
            raise RuntimeError(
                "input.size({}) must be equal to in_channels . Expected {}, got {}".format(
                    ch_dim, self.in_channels, input.size(ch_dim)
                )
            )

        if is_input_packed:
            mini_batch = int(batch_sizes[0])
        else:
            mini_batch = input.size(0) if self.batch_first else input.size(1)

        num_directions = 2 if self.bidirectional else 1
        expected_hidden_size = (
            self.num_layers * num_directions,
            mini_batch,
            self.out_channels,
        ) + input.shape[ch_dim + 1 :]

        def check_hidden_size(
            hx, expected_hidden_size, msg="Expected hidden size {}, got {}"
        ):
            if tuple(hx.size()) != expected_hidden_size:
                raise RuntimeError(msg.format(expected_hidden_size, tuple(hx.size())))

        if self.mode in ("LSTM", "PeepholeLSTM"):
            check_hidden_size(
                hidden[0], expected_hidden_size, "Expected hidden[0] size {}, got {}"
            )
            check_hidden_size(
                hidden[1], expected_hidden_size, "Expected hidden[1] size {}, got {}"
            )
        else:
            check_hidden_size(hidden, expected_hidden_size)

    def forward(self, input, hx=None):
        is_packed = isinstance(input, PackedSequence)
        if is_packed:
            input, batch_sizes = input
            max_batch_size = batch_sizes[0]
            insize = input.shape[2:]
        else:
            batch_sizes = None
            max_batch_size = input.size(0) if self.batch_first else input.size(1)
            insize = input.shape[3:]

        if hx is None:
            num_directions = 2 if self.bidirectional else 1
            hx = input.new_zeros(
                self.num_layers * num_directions,
                max_batch_size,
                self.out_channels,
                *insize,
                requires_grad=False
            )
            if self.mode in ("LSTM", "PeepholeLSTM"):
                hx = (hx, hx)

        self.check_forward_args(input, hx, batch_sizes)
        func = AutogradConvRNN(
            self.mode,
            num_layers=self.num_layers,
            batch_first=self.batch_first,
            dropout=self.dropout,
            train=self.training,
            bidirectional=self.bidirectional,
            variable_length=batch_sizes is not None,
            convndim=self.convndim,
            stride=self.stride,
            dilation=self.dilation,
            groups=self.groups,
        )
        output, hidden = func(input, self.all_weights, hx, batch_sizes)
        if is_packed:
            output = PackedSequence(output, batch_sizes)
        return output, hidden

    def extra_repr(self):
        s = (
            "{in_channels}, {out_channels}, kernel_size={kernel_size}"
            ", stride={stride}"
        )
        if self.dilation != (1,) * len(self.dilation):
            s += ", dilation={dilation}"
        if self.groups != 1:
            s += ", groups={groups}"
        if self.num_layers != 1:
            s += ", num_layers={num_layers}"
        if self.bias is not True:
            s += ", bias={bias}"
        if self.batch_first is not False:
            s += ", batch_first={batch_first}"
        if self.dropout != 0:
            s += ", dropout={dropout}"
        if self.bidirectional is not False:
            s += ", bidirectional={bidirectional}"
        return s.format(**self.__dict__)

    def __setstate__(self, d):
        super(ConvNdRNNBase, self).__setstate__(d)
        if "all_weights" in d:
            self._all_weights = d["all_weights"]
        if isinstance(self._all_weights[0][0], str):
            return
        num_layers = self.num_layers
        num_directions = 2 if self.bidirectional else 1
        self._all_weights = []
        for layer in range(num_layers):
            for direction in range(num_directions):
                suffix = "_reverse" if direction == 1 else ""
                if self.mode == "PeepholeLSTM":
                    weights = [
                        "weight_ih_l{}{}",
                        "weight_hh_l{}{}",
                        "weight_pi_l{}{}",
                        "weight_pf_l{}{}",
                        "weight_po_l{}{}",
                        "bias_ih_l{}{}",
                        "bias_hh_l{}{}",
                    ]
                else:
                    weights = [
                        "weight_ih_l{}{}",
                        "weight_hh_l{}{}",
                        "bias_ih_l{}{}",
                        "bias_hh_l{}{}",
                    ]
                weights = [x.format(layer, suffix) for x in weights]
                if self.bias:
                    self._all_weights += [weights]
                else:
                    self._all_weights += [weights[: len(weights) // 2]]

    @property
    def all_weights(self):
        return [
            [getattr(self, weight) for weight in weights]
            for weights in self._all_weights
        ]

class Conv2dGRU(ConvNdRNNBase):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Sequence[int]],
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: bool = False,
        stride: Union[int, Sequence[int]] = 1,
        dilation: Union[int, Sequence[int]] = 1,
        groups: int = 1,
    ):
        super().__init__(
            mode="GRU",
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional,
            convndim=2,
            stride=stride,
            dilation=dilation,
            groups=groups,
        )


def RNNReLUCell(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None, linear_func=None):
    """Copied from torch.nn._functions.rnn and modified"""
    if linear_func is None:
        linear_func = F.linear
    hy = F.relu(linear_func(input, w_ih, b_ih) + linear_func(hidden, w_hh, b_hh))
    return hy


def RNNTanhCell(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None, linear_func=None):
    """Copied from torch.nn._functions.rnn and modified"""
    if linear_func is None:
        linear_func = F.linear
    hy = torch.tanh(linear_func(input, w_ih, b_ih) + linear_func(hidden, w_hh, b_hh))
    return hy


def LSTMCell(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None, linear_func=None):
    """Copied from torch.nn._functions.rnn and modified"""
    if linear_func is None:
        linear_func = F.linear
    if input.is_cuda and linear_func is F.linear and fusedBackend is not None:
        igates = linear_func(input, w_ih)
        hgates = linear_func(hidden[0], w_hh)
        state = fusedBackend.LSTMFused.apply
        return (
            state(igates, hgates, hidden[1])
            if b_ih is None
            else state(igates, hgates, hidden[1], b_ih, b_hh)
        )

    hx, cx = hidden
    gates = linear_func(input, w_ih, b_ih) + linear_func(hx, w_hh, b_hh)
    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

    ingate = torch.sigmoid(ingate)
    forgetgate = torch.sigmoid(forgetgate)
    cellgate = torch.tanh(cellgate)
    outgate = torch.sigmoid(outgate)

    cy = (forgetgate * cx) + (ingate * cellgate)
    hy = outgate * torch.tanh(cy)

    return hy, cy


def PeepholeLSTMCell(
    input, hidden, w_ih, w_hh, w_pi, w_pf, w_po, b_ih=None, b_hh=None, linear_func=None
):
    if linear_func is None:
        linear_func = F.linear
    hx, cx = hidden
    gates = linear_func(input, w_ih, b_ih) + linear_func(hx, w_hh, b_hh)
    ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

    ingate += linear_func(cx, w_pi)
    forgetgate += linear_func(cx, w_pf)
    ingate = torch.sigmoid(ingate)
    forgetgate = torch.sigmoid(forgetgate)
    cellgate = torch.tanh(cellgate)

    cy = (forgetgate * cx) + (ingate * cellgate)
    outgate += linear_func(cy, w_po)
    outgate = torch.sigmoid(outgate)

    hy = outgate * torch.tanh(cy)

    return hy, cy


def GRUCell(input, hidden, w_ih, w_hh, b_ih=None, b_hh=None, linear_func=None):
    """Copied from torch.nn._functions.rnn and modified"""
    if linear_func is None:
        linear_func = F.linear
    if input.is_cuda and linear_func is F.linear and fusedBackend is not None:
        gi = linear_func(input, w_ih)
        gh = linear_func(hidden, w_hh)
        state = fusedBackend.GRUFused.apply
        return (
            state(gi, gh, hidden) if b_ih is None else state(gi, gh, hidden, b_ih, b_hh)
        )
    gi = linear_func(input, w_ih, b_ih)
    gh = linear_func(hidden, w_hh, b_hh)
    i_r, i_i, i_n = gi.chunk(3, 1)
    h_r, h_i, h_n = gh.chunk(3, 1)

    resetgate = torch.sigmoid(i_r + h_r)
    inputgate = torch.sigmoid(i_i + h_i)
    newgate = torch.tanh(i_n + resetgate * h_n)
    hy = newgate + inputgate * (hidden - newgate)

    return hy


def StackedRNN(inners, num_layers, lstm=False, dropout=0, train=True):
    """Copied from torch.nn._functions.rnn and modified"""

    num_directions = len(inners)
    total_layers = num_layers * num_directions

    def forward(input, hidden, weight, batch_sizes):
        assert len(weight) == total_layers
        next_hidden = []
        ch_dim = input.dim() - weight[0][0].dim() + 1

        if lstm:
            hidden = list(zip(*hidden))

        for i in range(num_layers):
            all_output = []
            for j, inner in enumerate(inners):
                l = i * num_directions + j

                hy, output = inner(input, hidden[l], weight[l], batch_sizes)
                next_hidden.append(hy)
                all_output.append(output)

            input = torch.cat(all_output, ch_dim)

            if dropout != 0 and i < num_layers - 1:
                input = F.dropout(input, p=dropout, training=train, inplace=False)

        if lstm:
            next_h, next_c = zip(*next_hidden)
            next_hidden = (
                torch.cat(next_h, 0).view(total_layers, *next_h[0].size()),
                torch.cat(next_c, 0).view(total_layers, *next_c[0].size()),
            )
        else:
            next_hidden = torch.cat(next_hidden, 0).view(
                total_layers, *next_hidden[0].size()
            )

        return next_hidden, input

    return forward


def Recurrent(inner, reverse=False):
    """Copied from torch.nn._functions.rnn without any modification"""

    def forward(input, hidden, weight, batch_sizes):
        output = []
        steps = range(input.size(0) - 1, -1, -1) if reverse else range(input.size(0))
        for i in steps:
            hidden = inner(input[i], hidden, *weight)
            # hack to handle LSTM
            output.append(hidden[0] if isinstance(hidden, tuple) else hidden)

        if reverse:
            output.reverse()
        output = torch.cat(output, 0).view(input.size(0), *output[0].size())

        return hidden, output

    return forward


def variable_recurrent_factory(inner, reverse=False):
    """Copied from torch.nn._functions.rnn without any modification"""
    if reverse:
        return VariableRecurrentReverse(inner)
    else:
        return VariableRecurrent(inner)


def VariableRecurrent(inner):
    """Copied from torch.nn._functions.rnn without any modification"""

    def forward(input, hidden, weight, batch_sizes):
        output = []
        input_offset = 0
        last_batch_size = batch_sizes[0]
        hiddens = []
        flat_hidden = not isinstance(hidden, tuple)
        if flat_hidden:
            hidden = (hidden,)
        for batch_size in batch_sizes:
            step_input = input[input_offset : input_offset + batch_size]
            input_offset += batch_size

            dec = last_batch_size - batch_size
            if dec > 0:
                hiddens.append(tuple(h[-dec:] for h in hidden))
                hidden = tuple(h[:-dec] for h in hidden)
            last_batch_size = batch_size

            if flat_hidden:
                hidden = (inner(step_input, hidden[0], *weight),)
            else:
                hidden = inner(step_input, hidden, *weight)

            output.append(hidden[0])
        hiddens.append(hidden)
        hiddens.reverse()

        hidden = tuple(torch.cat(h, 0) for h in zip(*hiddens))
        assert hidden[0].size(0) == batch_sizes[0]
        if flat_hidden:
            hidden = hidden[0]
        output = torch.cat(output, 0)

        return hidden, output

    return forward


def VariableRecurrentReverse(inner):
    """Copied from torch.nn._functions.rnn without any modification"""

    def forward(input, hidden, weight, batch_sizes):
        output = []
        input_offset = input.size(0)
        last_batch_size = batch_sizes[-1]
        initial_hidden = hidden
        flat_hidden = not isinstance(hidden, tuple)
        if flat_hidden:
            hidden = (hidden,)
            initial_hidden = (initial_hidden,)
        hidden = tuple(h[: batch_sizes[-1]] for h in hidden)
        for i in reversed(range(len(batch_sizes))):
            batch_size = batch_sizes[i]
            inc = batch_size - last_batch_size
            if inc > 0:
                hidden = tuple(
                    torch.cat((h, ih[last_batch_size:batch_size]), 0)
                    for h, ih in zip(hidden, initial_hidden)
                )
            last_batch_size = batch_size
            step_input = input[input_offset - batch_size : input_offset]
            input_offset -= batch_size

            if flat_hidden:
                hidden = (inner(step_input, hidden[0], *weight),)
            else:
                hidden = inner(step_input, hidden, *weight)
            output.append(hidden[0])

        output.reverse()
        output = torch.cat(output, 0)
        if flat_hidden:
            hidden = hidden[0]
        return hidden, output

    return forward


def ConvNdWithSamePadding(convndim=2, stride=1, dilation=1, groups=1):
    def forward(input, w, b=None):
        if convndim == 1:
            ntuple = _single
        elif convndim == 2:
            ntuple = _pair
        elif convndim == 3:
            ntuple = _triple
        else:
            raise ValueError("convndim must be 1, 2, or 3, but got {}".format(convndim))

        if input.dim() != convndim + 2:
            raise RuntimeError(
                "Input dim must be {}, bot got {}".format(convndim + 2, input.dim())
            )
        if w.dim() != convndim + 2:
            raise RuntimeError("w must be {}, bot got {}".format(convndim + 2, w.dim()))

        insize = input.shape[2:]
        kernel_size = w.shape[2:]
        _stride = ntuple(stride)
        _dilation = ntuple(dilation)

        ps = [
            (i + 1 - h + s * (h - 1) + d * (k - 1)) // 2
            for h, k, s, d in list(zip(insize, kernel_size, _stride, _dilation))[::-1]
            for i in range(2)
        ]
        # Padding to make the output shape to have the same shape as the input
        input = F.pad(input, ps, "constant", 0)
        return getattr(F, "conv{}d".format(convndim))(
            input,
            w,
            b,
            stride=_stride,
            padding=ntuple(0),
            dilation=_dilation,
            groups=groups,
        )

    return forward


def _conv_cell_helper(mode, convndim=2, stride=1, dilation=1, groups=1):
    linear_func = ConvNdWithSamePadding(
        convndim=convndim, stride=stride, dilation=dilation, groups=groups
    )

    if mode == "RNN_RELU":
        cell = partial(RNNReLUCell, linear_func=linear_func)
    elif mode == "RNN_TANH":
        cell = partial(RNNTanhCell, linear_func=linear_func)
    elif mode == "LSTM":
        cell = partial(LSTMCell, linear_func=linear_func)
    elif mode == "GRU":
        cell = partial(GRUCell, linear_func=linear_func)
    elif mode == "PeepholeLSTM":
        cell = partial(PeepholeLSTMCell, linear_func=linear_func)
    else:
        raise Exception("Unknown mode: {}".format(mode))
    return cell


def AutogradConvRNN(
    mode,
    num_layers=1,
    batch_first=False,
    dropout=0,
    train=True,
    bidirectional=False,
    variable_length=False,
    convndim=2,
    stride=1,
    dilation=1,
    groups=1,
):
    """Copied from torch.nn._functions.rnn and modified"""
    cell = _conv_cell_helper(
        mode, convndim=convndim, stride=stride, dilation=dilation, groups=groups
    )

    rec_factory = variable_recurrent_factory if variable_length else Recurrent

    if bidirectional:
        layer = (rec_factory(cell), rec_factory(cell, reverse=True))
    else:
        layer = (rec_factory(cell),)

    func = StackedRNN(
        layer,
        num_layers,
        (mode in ("LSTM", "PeepholeLSTM")),
        dropout=dropout,
        train=train,
    )

    def forward(input, weight, hidden, batch_sizes):
        if batch_first and batch_sizes is None:
            input = input.transpose(0, 1)

        nexth, output = func(input, hidden, weight, batch_sizes)

        if batch_first and batch_sizes is None:
            output = output.transpose(0, 1)

        return output, nexth

    return forward
