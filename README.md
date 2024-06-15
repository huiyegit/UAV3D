# UAV3D: A Large-scale 3D Perception Benchmark for Unmanned Aerial Vehicles


## Installation
- [Installation](https://github.com/huiyegit/UAV3D/tree/main)

## Dataset Download
Please check our [website](https://github.com/huiyegit/UAV3D/tree/main) to download the dataset.

After downloading the data, please put the data in the following structure:
```shell
├── data
│   ├── UAV3D
│   │   ├── maps
│   │   ├── samples
│   │   ├── sweeps
│   │   ├── v1.0-test
|   |   ├── v1.0-trainval
│   │   ├── uav3d_infos_train.pkl
│   │   ├── uav3d_infos_val.pkl
│   │   ├── uav3d_infos_test.pkl
```
## Train & inference
### Single UAV 3D object detection
#### Baseline PETR
```bash
cd perception/PETR
```
Training:
```bash
tools/dist_train.sh projects/configs/petr/petr_r50dcn_gridmask_p4.py 4 --work-dir work_dirs/petr_r50dcn_gridmask_p4/
```
Evaluation:
```bash
tools/dist_test.sh projects/configs/petr/petr_r50dcn_gridmask_p4.py work_dirs/petr_r50dcn_gridmask_p4/latest.pth 8 --eval bbox
```
#### Baseline BEVFusion
```bash
cd perception/bevfusion
```
Training:
```bash
torchpack dist-run -np 4  python tools/train.py configs/nuscenes/det/centerhead/lssfpn/camera/256x704/resnet/default.yaml    --run-dir runs/resnet50
```
Evaluation:
```bash
torchpack dist-run -np 4  python tools/test.py configs/nuscenes/det/centerhead/lssfpn/camera/256x704/resnet/default.yaml   runs/resnet50/epoch_24.pth   --eval bbox
```
#### Baseline DETR3D
```bash
cd perception/detr3d
```
Training:
```bash
tools/dist_train.sh     projects/configs/detr3d/detr3d_res50_gridmask.py   4  --work-dir      work_dirs/detr3d_res50_gridmask/
```
Evaluation:
```bash
tools/dist_test.sh      projects/configs/detr3d/detr3d_res50_gridmask.py     work_dirs/detr3d_res50_gridmask/epoch_24.pth  4  --eval bbox
```
### Collaborative UAVs 3D Object Detection
```bash
cd collaborative_perception/bevfusion
```
Training when2com(lowerbound / upperbound / v2vnet / when2com / who2com/ disonet):
```bash
torchpack dist-run -np 4  python tools/train.py configs/nuscenes/det/centerhead/lssfpn/camera/256x704/swint/when2com/default.yaml --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth    --run-dir runs/when2com
```
Evaluation of when2com(lowerbound / upperbound / v2vnet / when2com / who2com/ disonet):
```bash
torchpack dist-run -np 4  python tools/test.py configs/nuscenes/det/centerhead/lssfpn/camera/256x704/swint/when2com/default.yaml    runs/when2com/epoch_24.pth   --eval bbox
```
