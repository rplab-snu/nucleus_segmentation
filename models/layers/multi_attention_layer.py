import torch
import torch.nn as nn

from models.layers.unet_layer import weights_init_kaiming
from models.layers.grid_attention_layer import GridAttentionBlock2D, GridAttentionBlock3D, UnetGridGatingSignal2D, UnetGridGatingSignal3D

class MultiAttentionBlock2D(nn.Module):
    def __init__(self, in_size, gate_size, inter_size, nonlocal_mode, sub_sample_factor):
        super(MultiAttentionBlock2D, self).__init__()
        self.gate_block_1 = GridAttentionBlock2D(in_channels=in_size, gating_channels=gate_size,
                                                 inter_channels=inter_size, mode=nonlocal_mode,
                                                 sub_sample_factor= sub_sample_factor)
        self.gate_block_2 = GridAttentionBlock2D(in_channels=in_size, gating_channels=gate_size,
                                                 inter_channels=inter_size, mode=nonlocal_mode,
                                                 sub_sample_factor=sub_sample_factor)
        self.combine_gates = nn.Sequential(nn.Conv2d(in_size*2, in_size, kernel_size=1, stride=1, padding=0),
                                           nn.BatchNorm2d(in_size),
                                           nn.ReLU(inplace=True))

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('GridAttentionBlock3D') != -1:
                continue
            m.apply(weights_init_kaiming)

    def forward(self, input_, gating_signal):
        gate_1, att_1 = self.gate_block_1(input_, gating_signal)
        gate_2, att_2 = self.gate_block_2(input_, gating_signal)

        gate = torch.cat([gate_1, gate_2], 1)
        att  = torch.cat([att_1,  att_2],  1)
        return self.combine_gates(gate), att


class MultiAttentionBlock3D(nn.Module):
    def __init__(self, in_size, gate_size, inter_size, nonlocal_mode, sub_sample_factor):
        super(MultiAttentionBlock3D, self).__init__()
        self.gate_block_1 = GridAttentionBlock3D(in_channels=in_size, gating_channels=gate_size,
                                                 inter_channels=inter_size, mode=nonlocal_mode,
                                                 sub_sample_factor= sub_sample_factor)
        self.gate_block_2 = GridAttentionBlock3D(in_channels=in_size, gating_channels=gate_size,
                                                 inter_channels=inter_size, mode=nonlocal_mode,
                                                 sub_sample_factor=sub_sample_factor)
        self.combine_gates = nn.Sequential(nn.Conv3d(in_size*2, in_size, kernel_size=1, stride=1, padding=0),
                                           nn.BatchNorm3d(in_size),
                                           nn.ReLU(inplace=True))

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('GridAttentionBlock3D') != -1: 
                continue
            m.apply(weights_init_kaiming)
            

    def forward(self, input, gating_signal):
        gate_1, att_1 = self.gate_block_1(input, gating_signal)
        gate_2, att_2 = self.gate_block_2(input, gating_signal)

        gate = torch.cat([gate_1, gate_2], 1)
        att  = torch.cat([att_1,  att_2],  1)
        return self.combine_gates(gate), att
