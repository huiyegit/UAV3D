'''
Author: Pan Shi <shipan76@mail.ustc.edu.cn>
Date: 2023-12-27 14:43:11
LastEditTime: 2024-01-02 14:50:29
LastEditors: Pan Shi
Description: 
FilePath: /shipan/bevfusion/mmdet3d/models/builder.py
'''
from mmcv.utils import Registry

from mmdet.models.builder import BACKBONES, HEADS, LOSSES, NECKS

FUSIONMODELS = Registry("fusion_models")
VTRANSFORMS = Registry("vtransforms")
FUSERS = Registry("fusers")

WHEN2COM = Registry("when2com")
WHO2COM = Registry("who2com")
DISCONET = Registry("DiscoNet")
V2VNET = Registry("V2VNet")
LOWERBOUND = Registry("lowerbound")
MAXFUSION = Registry("MaxFusion")
SUMFUSION = Registry("SumFusion")
CATFUSION = Registry("CatFusion")
AGENTWISEWEIGHTEDFUSION = Registry("AgentWiseWeightedFusion")
def build_backbone(cfg):
    return BACKBONES.build(cfg)


def build_neck(cfg):
    return NECKS.build(cfg)


def build_vtransform(cfg):
    return VTRANSFORMS.build(cfg)


def build_fuser(cfg):
    return FUSERS.build(cfg)


def build_head(cfg):
    return HEADS.build(cfg)


def build_loss(cfg):
    return LOSSES.build(cfg)


def build_fusion_model(cfg, train_cfg=None, test_cfg=None):
    return FUSIONMODELS.build(
        cfg, default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg)
    )


def build_model(cfg, train_cfg=None, test_cfg=None):
    return build_fusion_model(cfg, train_cfg=train_cfg, test_cfg=test_cfg)
