## Training and evaluation
### Single UAV 3D object detection
#### Baseline PETR
```bash
cd perception/PETR
```
Training(image size  <b>800x450</b>):
```bash
tools/dist_train.sh projects/configs/petr/petr_r50dcn_800_450.py 4 --work-dir work_dirs/petr_r50dcn/
```
Evaluation(image size  <b>800x450</b>):
```bash
tools/dist_test.sh projects/configs/petr/petr_r50dcn_800_450.py work_dirs/petr_r50dcn/epoch_24.pth 4 --eval bbox
```

Training(image size <b>704x256</b>):

* Uncomment line 71 in [nuscenes_dataset.py](https://github.com/huiyegit/UAV3D/blob/main/perception/PETR/projects/mmdet3d_plugin/datasets/nuscenes_dataset.py)
* Uncomment line 111 in [transform_3d.py](https://github.com/huiyegit/UAV3D/blob/main/perception/PETR/projects/mmdet3d_plugin/datasets/pipelines/transform_3d.py)

```bash
tools/dist_train.sh projects/configs/petr/petr_r50dcn_704_256.py 4 --work-dir work_dirs/petr_r50dcn/
```

Evaluation(image size <b>704x256</b>):
```bash
tools/dist_test.sh projects/configs/petr/petr_r50dcn_704_256.py work_dirs/petr_r50dcn/epoch_24.pth 4 --eval bbox
```
#### Baseline BEVFusion
```bash
cd perception/bevfusion
```
Training(image size  <b>800x450</b>):
```bash
torchpack dist-run -np 4  python tools/train.py configs/nuscenes/det/centerhead/lssfpn/camera/450x800/resnet/default.yaml    --run-dir runs/resnet101
```
Evaluation(image size  <b>800x450</b>):
```bash
torchpack dist-run -np 4  python tools/test.py configs/nuscenes/det/centerhead/lssfpn/camera/450x800/resnet/default.yaml   runs/resnet101/epoch_24.pth   --eval bbox
```


Training(image size <b>704x256</b>):
```bash
torchpack dist-run -np 4  python tools/train.py configs/nuscenes/det/centerhead/lssfpn/camera/256x704/resnet/default.yaml    --run-dir runs/resnet50
```
Evaluation(image size <b>704x256</b>):
```bash
torchpack dist-run -np 4  python tools/test.py configs/nuscenes/det/centerhead/lssfpn/camera/256x704/resnet/default.yaml   runs/resnet50/epoch_24.pth   --eval bbox
```
#### Baseline DETR3D
```bash
cd perception/detr3d
```
Training(image size  <b>800x450</b>):
```bash
tools/dist_train.sh     projects/configs/detr3d/detr3d_res101_gridmask.py   4  --work-dir      work_dirs/detr3d_res101_gridmask/
```
Evaluation(image size  <b>800x450</b>):
```bash
tools/dist_test.sh      projects/configs/detr3d/detr3d_res101_gridmask.py     work_dirs/detr3d_res101_gridmask/epoch_24.pth  4  --eval bbox
```
Training(image size <b>704x256</b>):

* Uncomment line 243 in [nuscenes_dataset.py](https://github.com/huiyegit/UAV3D/blob/main/perception/detr3d/mmdetection3d/mmdet3d/datasets/nuscenes_dataset.py)
* Uncomment line 89 in [transform_3d.py](https://github.com/huiyegit/UAV3D/blob/main/perception/detr3d/projects/mmdet3d_plugin/datasets/pipelines/transform_3d.py)

```bash
tools/dist_train.sh     projects/configs/detr3d/detr3d_res50_gridmask.py   4  --work-dir      work_dirs/detr3d_res50_gridmask/
```
Evaluation(image size <b>704x256</b>):
```bash
tools/dist_test.sh      projects/configs/detr3d/detr3d_res50_gridmask.py     work_dirs/detr3d_res50_gridmask/epoch_24.pth  4  --eval bbox
```


### Collaborative UAVs 3D Object Detection
```bash
cd collaborative_perception/bevfusion
```
Training <b>when2com</b>(lowerbound / upperbound / v2vnet / when2com / who2com/ disconet):
```bash
torchpack dist-run -np 4  python tools/train.py configs/nuscenes/det/centerhead/lssfpn/camera/256x704/swint/when2com/default.yaml --model.encoders.camera.backbone.init_cfg.checkpoint pretrained/swint-nuimages-pretrained.pth    --run-dir runs/when2com
```
Evaluation of <b>when2com</b>(lowerbound / upperbound / v2vnet / when2com / who2com/ disconet):
```bash
torchpack dist-run -np 4  python tools/test.py configs/nuscenes/det/centerhead/lssfpn/camera/256x704/swint/when2com/default.yaml    runs/when2com/epoch_24.pth   --eval bbox
```
### Object Tracking

```bash
cd tracking/CenterPoint
```
Put the `results_nusc.json` file in the  `CenterPoint/input` folder.

Evaluation:
```bash
python pub_test.py --work_dir /{dir}/UAV3D/tracking/CenterPoint/output  --checkpoint    /{dir}/UAV3D/tracking/CenterPoint/input/results_nusc.json 
```

