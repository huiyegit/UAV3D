'''
Author: Pan Shi <shipan76@mail.ustc.edu.cn>
Date: 2023-12-16 15:43:20
LastEditTime: 2023-12-20 19:55:47
LastEditors: Pan Shi
Description: 
FilePath: /shipan/bevfusion-main/mmdet3d/models/com/DiscoNet.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .Backbone import *
class DiscoNet(nn.Module):
    """DiscoNet.

    https://github.com/ai4ce/DiscoNet

    Args:
        config (object): The config object.
        layer (int, optional): Collaborate on which layer. Defaults to 3.
        in_channels (int, optional): The input channels. Defaults to 13.
        kd_flag (bool, optional): Whether to use knowledge distillation. Defaults to True.
        num_agent (int, optional): The number of agents (including RSU). Defaults to 5.

    """

    def __init__(self, config, layer=3, in_channels=13, kd_flag=True, num_agent=5, compress_level=0, only_v2i=False, p_com_outage = 0.0):
        super().__init__()
        self.p_com_outage = p_com_outage
        self.only_v2i = only_v2i

        self.pixel_weighted_fusion = PixelWeightedFusionSoftmax(config['in_channels'])
        
        self.layer_channel = config['in_channels']
        self.downsample = Conv2DBatchNormRelu(self.layer_channel, self.layer_channel, k_size=3, stride=2, padding=1)
        # self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        
        self.downsample_2 = Conv2DBatchNormRelu(self.layer_channel, self.layer_channel, k_size=3, stride=2, padding=1)
        # self.upsample_2 = nn.Upsample(scale_factor=2, mode='nearest')
        
        self.downsample_3 = Conv2DBatchNormRelu(self.layer_channel, self.layer_channel, k_size=3, stride=2, padding=1)
        # self.upsample_3 = nn.Upsample(scale_factor=2, mode='nearest')
        
        self.downsample_4 = Conv2DBatchNormRelu(self.layer_channel, self.layer_channel, k_size=3, stride=2, padding=1)
        # self.upsample_4 = nn.Upsample(scale_factor=2, mode='nearest')
        
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
        

    def outage(self) -> bool:
        """Simulate communication outage according to self.p_com_outage.

        Returns:
            A bool indicating if the communication outage happens.
        """
        return np.random.choice(
            [True, False], p=[self.p_com_outage, 1 - self.p_com_outage]
        )


    def forward(self, x, training=True):
        """Forward pass.

        Args:
            bevs (tensor): BEV data
            trans_matrices (tensor): Matrix for transforming features among agents.
            num_agent_tensor (tensor): Number of agents to communicate for each agent.
            batch_size (int, optional): The batch size. Defaults to 1.

        Returns:
            result, all decoded layers, and fused feature maps if kd_flag is set.
            else return result and list of weights for each agent.
        """


        # local_com_mat = torch.cat(tuple(feat_list), 1)  # [2 5 512 16 16] [batch, agent, channel, height, width]
        # local_com_mat_update = super().build_local_communication_matrix(
        #     feat_list
        # )  # to avoid the inplace operation
        B, N, C, H, W = x.size()
        
        x = x.view(B * N, C, H, W)
        x = self.downsample(x)
        x = self.downsample_2(x)
        # x = self.downsample_3(x)
        # x = self.downsample_4(x)
        
        _, C, H, W = x.size()
        x = x.view(B, N, C, H, W)
        
        
        local_com_mat_update = x.clone()
        for b in range(B):
            num_agent = N
            for i in range(num_agent):
                tg_agent = x[b, i]
                # all_warp = trans_matrices[b, i]  # transformation [2 5 5 4 4]

                self.neighbor_feat_list = list()
                self.neighbor_feat_list.append(tg_agent)

                if self.outage():
                    agent_wise_weight_feat = self.neighbor_feat_list[0]
                else:
                    for j in range(num_agent):
                        if j != i:
                            if self.only_v2i and i != 0 and j != 0:
                                continue
                            warp_feat = x[b, j]
                            self.neighbor_feat_list.append(warp_feat)
                    # agent-wise weighted fusion
                    tmp_agent_weight_list = list()
                    sum_weight = 0
                    nb_len = len(self.neighbor_feat_list)
                    for k in range(nb_len):
                        cat_feat = torch.cat(
                            [tg_agent, self.neighbor_feat_list[k]], dim=0
                        )
                        cat_feat = cat_feat.unsqueeze(0)
                        agent_weight = torch.squeeze(
                            self.pixel_weighted_fusion(cat_feat)
                        )
                        tmp_agent_weight_list.append(torch.exp(agent_weight))
                        sum_weight = sum_weight + torch.exp(agent_weight)

                    agent_weight_list = list()
                    for k in range(nb_len):
                        agent_weight = torch.div(tmp_agent_weight_list[k], sum_weight)
                        agent_weight.expand([256, -1, -1])
                        agent_weight_list.append(agent_weight)

                    agent_wise_weight_feat = 0
                    for k in range(nb_len):
                        agent_wise_weight_feat = (
                            agent_wise_weight_feat
                            + agent_weight_list[k] * self.neighbor_feat_list[k]
                        )

                # feature update
                local_com_mat_update[b, i] = agent_wise_weight_feat

        # import pdb
        # pdb.set_trace()
        # x = torch.mean(local_com_mat_update, dim=1)
        
        B, N, C, H, W = local_com_mat_update.size()
        local_com_mat_update = local_com_mat_update.view(B * N, C, H, W)
        local_com_mat_update = self.upsample(local_com_mat_update)
        local_com_mat_update = self.upsample_2(local_com_mat_update)
        # local_com_mat_update = self.upsample_3(local_com_mat_update)
        # local_com_mat_update = self.upsample_4(local_com_mat_update)
        
        _, C, H, W = local_com_mat_update.size()
        local_com_mat_update = local_com_mat_update.view(B, N, C, H, W)
        
        
        return  local_com_mat_update
        # return  local_com_mat_update.unbind(dim=1)[0]

    



class PixelWeightedFusionSoftmax(nn.Module):
    def __init__(self, channel):
        super(PixelWeightedFusionSoftmax, self).__init__()

        self.conv1_1 = nn.Conv2d(channel * 2, 128, kernel_size=1, stride=1, padding=0)
        self.bn1_1 = nn.BatchNorm2d(128)

        self.conv1_2 = nn.Conv2d(128, 32, kernel_size=1, stride=1, padding=0)
        self.bn1_2 = nn.BatchNorm2d(32)

        self.conv1_3 = nn.Conv2d(32, 8, kernel_size=1, stride=1, padding=0)
        self.bn1_3 = nn.BatchNorm2d(8)

        self.conv1_4 = nn.Conv2d(8, 1, kernel_size=1, stride=1, padding=0)
        # self.bn1_4 = nn.BatchNorm2d(1)

    def forward(self, x):
        x = x.view(-1, x.size(-3), x.size(-2), x.size(-1))
        x_1 = F.relu(self.bn1_1(self.conv1_1(x)))
        x_1 = F.relu(self.bn1_2(self.conv1_2(x_1)))
        x_1 = F.relu(self.bn1_3(self.conv1_3(x_1)))
        x_1 = F.relu(self.conv1_4(x_1))

        return x_1
