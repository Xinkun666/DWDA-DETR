# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from util.misc import NestedTensor, is_main_process

from models.position_encoding import build_position_encoding
from models.swin_transformer import SwinTransformer
from models.dat import DAT

class Backbone():
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d)
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048



class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list.tensors)
        out: List[NestedTensor] = []
        pos = []
        out.append(xs)
        pos.append(self[1](xs).to(xs.tensors.dtype))
        # for name, x in xs.items():
        #     out.append(x)
        #     # position encoding
        #     pos.append(self[1](x).to(x.tensors.dtype))
        return out, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    backbone = DAT()
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model
