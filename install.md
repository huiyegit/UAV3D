
## Conda environment

### BEVFusion
Option 1 : Follow the installation instructions of [BEVFusion](https://github.com/mit-han-lab/bevfusion). If some versions of packages are not provided, please refer to our [requirements](./requirements_bevfusion.txt) for BEVFusion. Note: We do not use the latest source code of BEVFusion repo. Please use the [version](./perception/bevfusion) for UAV3D. 

### PETR
Option 1 : Follow the installation instructions of [PETR](https://github.com/megvii-research/PETR). If some versions of packages are not provided, please refer to our [requirements](./requirements_petr.txt) for PETR.


### DETR3D
Option 1 : Follow the installation instructions of [DETR3D](https://github.com/WangYueFt/detr3d). If some versions of packages are not provided, please refer to our [requirements](./requirements_detr3d.txt) for DETR3D.

## 
```bash
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html
```
examples：
```bash
pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
```
## Install MMDetection

```bash
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection
git checkout v2.24.1 
sudo pip install -r requirements/build.txt
sudo python3 setup.py develop
cd ..
```

## Install MMSegmentation.

```bash
sudo pip install mmsegmentation==0.20.2
```

## Install MMDetection3D

```bash
git clone  https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout v0.17.1 
sudo pip install -r requirements/build.txt
sudo python3 setup.py develop
cd ..
```

## Install PETR

```bash
git clone https://github.com/megvii-research/PETR.git
cd PETR
mkdir ckpts
mkdir data
ln -s {mmdetection3d_path} ./mmdetection3d
ln -s {nuscenes_path} ./data/nuscenes
```
examples
```bash
git clone https://github.com/megvii-research/PETR.git
cd PETR
mkdir ckpts ###pretrain weights
mkdir data ###dataset
ln -s ../mmdetection3d ./mmdetection3d
ln -s /data/Dataset/nuScenes ./data/nuscenes
```





