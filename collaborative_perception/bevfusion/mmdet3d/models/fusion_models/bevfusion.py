from typing import Any, Dict

import torch
from mmcv.runner import auto_fp16, force_fp32
from torch import nn
from torch.nn import functional as F

from mmdet3d.models.builder import (
    build_backbone,
    build_fuser,
    build_head,
    build_neck,
    build_vtransform,
)
from mmdet3d.ops import Voxelization, DynamicScatter
from mmdet3d.models import FUSIONMODELS

from mmdet3d.models.com import *
from .base import Base3DFusionModel

__all__ = ["BEVFusion"]


@FUSIONMODELS.register_module()
class BEVFusion(Base3DFusionModel):
    def __init__(
        self,
        encoders: Dict[str, Any],
        fuser: Dict[str, Any],
        decoder: Dict[str, Any],
        heads: Dict[str, Any],
        **kwargs,
    ) -> None:
        super().__init__()

        self.encoders = nn.ModuleDict()
        if encoders.get("camera") is not None:
            self.encoders["camera"] = nn.ModuleDict(
                {
                    "backbone": build_backbone(encoders["camera"]["backbone"]),
                    "neck": build_neck(encoders["camera"]["neck"]),
                    "vtransform": build_vtransform(encoders["camera"]["vtransform"]),
                    "com":None,
                }
            )
        # self.teacher = False
        self.com_type = encoders["camera"]["com"]["type"]
        if self.com_type == "when2com":
            self.encoders["camera"]["com"] = When2com(encoders["camera"]["com"])
        if self.com_type == "who2com":
            self.encoders["camera"]["com"] = Who2com(encoders["camera"]["com"])
        elif self.com_type == "DiscoNet":
            self.encoders["camera"]["com"] = DiscoNet(encoders["camera"]["com"])
            self.teacher = TeacherNet(encoders,fuser, decoder, heads, **kwargs,)
            # for k, v in self.teacher.named_parameters():
                # v.requires_grad = False  # fix parameters
            # self.teacher = True
            # self.teacher_layer = []
            # self.student_layer = []
        elif self.com_type == "V2VNet":
            self.encoders["camera"]["com"] = V2VNet(encoders["camera"]["com"])
        elif self.com_type == "lowerbound":
            pass
        elif self.com_type == "upperbound":
            pass
        elif self.com_type == "MaxFusion":
            self.encoders["camera"]["com"] = MaxFusion(encoders["camera"]["com"])
        elif self.com_type == "SumFusion":
            self.encoders["camera"]["com"] = SumFusion(encoders["camera"]["com"])
        elif self.com_type == "CatFusion":
            self.encoders["camera"]["com"] = CatFusion(encoders["camera"]["com"])
        elif self.com_type == "AgentWiseWeightedFusion":
            self.encoders["camera"]["com"] = AgentWiseWeightedFusion(encoders["camera"]["com"])
        else:
            pass
        if encoders["camera"]["com"].get("agent_num") is not None:
            self.agent_num = encoders["camera"]["com"]["agent_num"]
        # self.com_type = "DiscoNet"
        if encoders.get("lidar") is not None:
            if encoders["lidar"]["voxelize"].get("max_num_points", -1) > 0:
                voxelize_module = Voxelization(**encoders["lidar"]["voxelize"])
            else:
                voxelize_module = DynamicScatter(**encoders["lidar"]["voxelize"])
            self.encoders["lidar"] = nn.ModuleDict(
                {
                    "voxelize": voxelize_module,
                    "backbone": build_backbone(encoders["lidar"]["backbone"]),
                }
            )
            self.voxelize_reduce = encoders["lidar"].get("voxelize_reduce", True)

        if fuser is not None:
            self.fuser = build_fuser(fuser)
        else:
            self.fuser = None

        self.decoder = nn.ModuleDict(
            {
                "backbone": build_backbone(decoder["backbone"]),
                "neck": build_neck(decoder["neck"]),
            }
        )
        self.heads = nn.ModuleDict()
        for name in heads:
            if heads[name] is not None:
                self.heads[name] = build_head(heads[name])

        if "loss_scale" in kwargs:
            self.loss_scale = kwargs["loss_scale"]
        else:
            self.loss_scale = dict()
            for name in heads:
                if heads[name] is not None:
                    self.loss_scale[name] = 1.0

        self.init_weights()

    def init_weights(self) -> None:
        if "camera" in self.encoders:
            self.encoders["camera"]["backbone"].init_weights()

    def extract_camera_features(
        self,
        x,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        img_metas,
    ) -> torch.Tensor:
        B, N, C, H, W = x.size()
        x = x.view(B * N, C, H, W)

        x = self.encoders["camera"]["backbone"](x)        
        x = self.encoders["camera"]["neck"](x)

        if not isinstance(x, torch.Tensor):
            x = x[0]
            # x = x[1]

        BN, C, H, W = x.size()
        x = x.view(B, int(BN / B), C, H, W)

        x = self.encoders["camera"]["vtransform"](
            x,
            points,
            camera2ego,
            lidar2ego,
            lidar2camera,
            lidar2image,
            camera_intrinsics,
            camera2lidar,
            img_aug_matrix,
            lidar_aug_matrix,
            img_metas,
        )
        
        n_cam = 5
        x_splits = torch.split(x, n_cam, dim = 1)
        x_temp = [ torch.mean(split, dim=1) for split in x_splits]
        x = torch.stack(x_temp,dim=1)
        
        if self.com_type in ["lowerbound", "upperbound"]:
            x_com = x
        else:
            x_com = self.encoders["camera"]['com'](x,training=self.training)
            
        if self.com_type in ["upperbound"]:
            n_agent = 5
        elif self.com_type in ["V2VNet", "DiscoNet", "lowerbound","when2com", "who2com"]:
            n_agent = 1
        
        
        x = x_com[:,0:n_agent,:]
        x = torch.mean(x, dim=1)
        
        x_fuse = x
        
        # B, N, C, H, W = x.size()
        # x = x.view(B * N, C, H, W)
        x = self.encoders["camera"]["vtransform"].downsample(x)
        

        return x, x_fuse

    def extract_lidar_features(self, x) -> torch.Tensor:
        feats, coords, sizes = self.voxelize(x)
        batch_size = coords[-1, 0] + 1
        x = self.encoders["lidar"]["backbone"](feats, coords, batch_size, sizes=sizes)
        return x

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        feats, coords, sizes = [], [], []
        for k, res in enumerate(points):
            ret = self.encoders["lidar"]["voxelize"](res)
            if len(ret) == 3:
                # hard voxelize
                f, c, n = ret
            else:
                assert len(ret) == 2
                f, c = ret
                n = None
            feats.append(f)
            coords.append(F.pad(c, (1, 0), mode="constant", value=k))
            if n is not None:
                sizes.append(n)

        feats = torch.cat(feats, dim=0)
        coords = torch.cat(coords, dim=0)
        if len(sizes) > 0:
            sizes = torch.cat(sizes, dim=0)
            if self.voxelize_reduce:
                feats = feats.sum(dim=1, keepdim=False) / sizes.type_as(feats).view(
                    -1, 1
                )
                feats = feats.contiguous()

        return feats, coords, sizes

    @auto_fp16(apply_to=("img", "points"))
    def forward(
        self,
        img,
        # points=None,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        points=None,
        gt_masks_bev=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        **kwargs,
    ):
        if isinstance(img, list):
            raise NotImplementedError
        else:
            outputs = self.forward_single(
                img,
                points,
                camera2ego,
                lidar2ego,
                lidar2camera,
                lidar2image,
                camera_intrinsics,
                camera2lidar,
                img_aug_matrix,
                lidar_aug_matrix,
                metas,
                gt_masks_bev,
                gt_bboxes_3d,
                gt_labels_3d,
                **kwargs,
            )
            return outputs

    @auto_fp16(apply_to=("img", "points"))
    def forward_single(
        self,
        img,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        gt_masks_bev=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        **kwargs,
    ):
        features = []
        for sensor in (
            self.encoders if self.training else list(self.encoders.keys())[::-1]
        ):
            if sensor == "camera":
                feature, x_fuse = self.extract_camera_features(
                    img,
                    points,
                    camera2ego,
                    lidar2ego,
                    lidar2camera,
                    lidar2image,
                    camera_intrinsics,
                    camera2lidar,
                    img_aug_matrix,
                    lidar_aug_matrix,
                    metas,
                )
            elif sensor == "lidar":
                feature = self.extract_lidar_features(points)
            else:
                raise ValueError(f"unsupported sensor: {sensor}")
            features.append(feature)

        if not self.training:
            # avoid OOM
            features = features[::-1]

        if self.fuser is not None:
            x = self.fuser(features)
        else:
            assert len(features) == 1, features
            x = features[0]

        batch_size = x.shape[0]

        x = self.decoder["backbone"](x)
        x = self.decoder["neck"](x)

        if self.training:
            outputs = {}
            for type, head in self.heads.items():
                if type == "object":
                    pred_dict = head(x, metas)
                    losses = head.loss(gt_bboxes_3d, gt_labels_3d, pred_dict)
                elif type == "map":
                    losses = head(x, gt_masks_bev)
                else:
                    raise ValueError(f"unsupported head: {type}")
                
                if self.com_type == "DiscoNet":
                    (
                        x_t1
                    ) = self.teacher(img,
                    points,
                    camera2ego,
                    lidar2ego,
                    lidar2camera,
                    lidar2image,
                    camera_intrinsics,
                    camera2lidar,
                    img_aug_matrix,
                    lidar_aug_matrix,
                    metas,
                    )
                    kd_loss = self.get_kd_loss(batch_size,self.agent_num,x_t1, x_fuse)
                    losses["kd_loss"] = kd_loss
                    

                for name, val in losses.items():
                    if val.requires_grad:
                        if val == "kd_loss":
                            outputs[f"stats/{type}/{name}"] = val
                        else:
                            outputs[f"loss/{type}/{name}"] = val * self.loss_scale[type]
                    else:
                        outputs[f"stats/{type}/{name}"] = val
            return outputs
        else:
            outputs = [{} for _ in range(batch_size)]
            for type, head in self.heads.items():
                if type == "object":
                    pred_dict = head(x, metas)
                    bboxes = head.get_bboxes(pred_dict, metas)
                    for k, (boxes, scores, labels) in enumerate(bboxes):
                        outputs[k].update(
                            {
                                "boxes_3d": boxes.to("cpu"),
                                "scores_3d": scores.cpu(),
                                "labels_3d": labels.cpu(),
                            }
                        )
                elif type == "map":
                    logits = head(x)
                    for k in range(batch_size):
                        outputs[k].update(
                            {
                                "masks_bev": logits[k].cpu(),
                                "gt_masks_bev": gt_masks_bev[k].cpu(),
                            }
                        )
                else:
                    raise ValueError(f"unsupported head: {type}")
            return outputs
        
    def get_kd_loss(self, batch_size, num_agent, x_t1, x_fuse, kd_weight=100000):
        
        
        # for k, v in self.teacher.named_parameters():
        # 	if k != 'xxx.weight' and k != 'xxx.bias':
        # 		print(v.requires_grad)  # should be False

        # for k, v in self.model.named_parameters():
        # 	if k != 'xxx.weight' and k != 'xxx.bias':
        # 		print(v.requires_grad)  # should be False

        # -------- KD loss---------#
        kl_loss_mean = nn.KLDivLoss(reduction='mean')
        # BN,C,H,W = teacher_encode.size()
        # teacher_encode = teacher_encode.view(batch_size,num_agent,C,H,W)
        # BN,C,H,W = teacher_decode.size()
        # teacher_decode = teacher_decode.view(batch_size,num_agent,C,H,W)
        # teacher_encode = torch.max(teacher_encode, dim=1).values
        # teacher_decode = torch.max(teacher_decode, dim=1).values
        
        # num_agent = 2
        
        BN,C,H,W = x_t1.size()
        # x_t1 = x_t1.view(batch_size, -1, C, H, W)
        # x_t1 = torch.mean(x_t1, dim=1)
        # BN,C,H,W = x_t1.size()
        
        target_x1 = x_t1.permute(0, 2, 3, 1).reshape(
            BN * H * W, -1
        )
        
        BN,C,H,W = x_fuse.size()
        # x_fuse = x_fuse.view(batch_size, -1, C, H, W)
        # x_fuse = x_fuse[:,0:1,:].squeeze(dim=1)
        # BN,C,H,W = x_fuse.size()
        student_x1 = x_fuse.permute(0, 2, 3, 1).reshape(
            BN * H * W, -1
        )
        kd_loss_x1 = kl_loss_mean(
                F.log_softmax(student_x1, dim=1), F.softmax(target_x1, dim=1)
            )

        kd_weight = 100000
        # kd_weight = 10000
        # kd_weight = 0
        kd_loss = kd_weight * kd_loss_x1
        # kd_loss = kd_weight * (kd_loss_x6 + kd_loss_x5 + kd_loss_fused_layer)
        # print(kd_loss)
        return kd_loss
    
class TeacherNet(Base3DFusionModel):
    """The teacher net for knowledged distillation in DiscoNet."""

    def __init__(
        self,
        encoders: Dict[str, Any],
        fuser: Dict[str, Any],
        decoder: Dict[str, Any],
        heads: Dict[str, Any],
        **kwargs,
    ) -> None:
        super().__init__()
        self.encoders = nn.ModuleDict()

        self.encoders["camera"] = nn.ModuleDict(
            {
                "backbone": build_backbone(encoders["camera"]["backbone"]),
                "neck": build_neck(encoders["camera"]["neck"]),
                "vtransform": build_vtransform(encoders["camera"]["vtransform"]),
                "com":None,
            }
        )
        
        self.decoder = nn.ModuleDict(
            {
                "backbone": build_backbone(decoder["backbone"]),
                "neck": build_neck(decoder["neck"]),
            }
        )
        self.heads = nn.ModuleDict()
        for name in heads:
            if heads[name] is not None:
                self.heads[name] = build_head(heads[name])

        if "loss_scale" in kwargs:
            self.loss_scale = kwargs["loss_scale"]
        else:
            self.loss_scale = dict()
            for name in heads:
                if heads[name] is not None:
                    self.loss_scale[name] = 1.0

        self.init_weights()

    def init_weights(self) -> None:
        if "camera" in self.encoders:
            self.encoders["camera"]["backbone"].init_weights()  
        


    def forward(self, x,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        img_metas,
        **kwargs,):
        B, N, C, H, W = x.size()
        x = x.view(B * N, C, H, W)

        x = self.encoders["camera"]["backbone"](x)
        
        # x_t1 = x[1] 
        
        x = self.encoders["camera"]["neck"](x)
        
        # x_t3, x_t4 = x[0], x[1]

        if not isinstance(x, torch.Tensor):
            x = x[0]

        BN, C, H, W = x.size()
        x = x.view(B, int(BN / B), C, H, W)
        x = self.encoders["camera"]["vtransform"](
                x,
                points,
                camera2ego,
                lidar2ego,
                lidar2camera,
                lidar2image,
                camera_intrinsics,
                camera2lidar,
                img_aug_matrix,
                lidar_aug_matrix,
                img_metas,
            )
        # x = torch.cat(x.unbind(dim=1), 0)
        
        x = torch.mean(x, dim=1)
        x_t1 = x
        
        encode = self.encoders["camera"]["vtransform"].downsample(x)
        
        # decode = self.decoder["backbone"](encode)
        # decode = self.decoder["neck"](decode)

        return x_t1
