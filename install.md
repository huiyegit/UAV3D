
## Conda environment

### BEVFusion
**Option 1** : Follow the installation instructions of [BEVFusion](https://github.com/mit-han-lab/bevfusion). If some versions of packages are not provided, please refer to our [requirements](./requirements_bevfusion.txt) for BEVFusion. **Note**: We do not use the latest source code of __BEVFusion__ repo. Please use this [version](./perception/bevfusion) for __UAV3D__. 

**Option 2** : 

conda create --name bevfusion  python=3.8

conda activate bevfusion

conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch
pip install Pillow==8.4.0
pip install tqdm
pip install torchpack
pip install mmcv==1.4.0 mmcv-full==1.4.0 mmdet==2.20.0
pip install nuscenes-devkit
pip install mpi4py==3.0.3
pip install numba==0.48.0

pip uninstall numpy && pip install numpy==1.22.3
pip uninstall yapf && pip install yapf==0.40.1
pip uninstall setuptools && pip install setuptools==59.5.0
pip install  transforms3d




### PETR
**Option 1** : Follow the installation instructions of [PETR](https://github.com/megvii-research/PETR). If some versions of packages are not provided, please refer to our [requirements](./requirements_petr.txt) for PETR.


### DETR3D
**Option 1** : Follow the installation instructions of [DETR3D](https://github.com/WangYueFt/detr3d). If some versions of packages are not provided, please refer to our [requirements](./requirements_detr3d.txt) for DETR3D.

## Docker environment
**Option 1** : Follow the installation instructions of [BEVFusion](https://github.com/mit-han-lab/bevfusion). Then follow the instructions __above__ to install the Conda enviroment in Docker. 


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





