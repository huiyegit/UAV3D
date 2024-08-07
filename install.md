
## Conda environment

### BEVFusion
**Option 1** : Follow the installation instructions of [BEVFusion](https://github.com/mit-han-lab/bevfusion). If some versions of packages are not provided, please refer to our [requirements](./requirements_bevfusion.txt) for BEVFusion. **Note**: We do not use the latest source code of __BEVFusion__ repo. Please use this [version](./perception/bevfusion) for __UAV3D__. 

**Option 2** : 
```bash
conda create --name bevfusion  python=3.8
conda activate bevfusion
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch
```
```bash
pip install Pillow==8.4.0
pip install tqdm==4.66.2
pip install torchpack==0.3.1
pip install mmcv==1.4.0 mmcv-full==1.4.0 mmdet==2.20.0
pip install nuscenes-devkit==1.1.11
pip install mpi4py==3.0.3
pip install numba==0.48.0
```
```bash
pip uninstall numpy && pip install numpy==1.22.3
pip uninstall yapf && pip install yapf==0.40.1
pip uninstall setuptools && pip install setuptools==59.5.0
pip install  transforms3d==0.4.1
```
```bash
git clone https://github.com/huiyegit/UAV3D.git
cd UAV3D/perception/bevfusion
python setup.py develop
```



### PETR
**Option 1** : Follow the installation instructions of [PETR](https://github.com/megvii-research/PETR). If some versions of packages are not provided, please refer to our [requirements](./requirements_petr.txt) for PETR.

**Option 2** : 
```bash
conda create --name petr python=3.8
conda activate petr
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch
```

```bash
pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.1/index.html
pip install mmdet==2.24.1
pip install mmsegmentation==0.20.2
```
```bash
cd UAV3D/perception/PETR/mmdetection3d
python3 setup.py develop  
```

```bash
pip install scikit-image==0.19.3
pip uninstall numba && pip install numba==0.48.0
pip install  nuscenes-devkit==1.1.10
pip install  lyft_dataset_sdk==0.0.8
pip install  plyfile==1.0.3
pip uninstall networkx && pip install networkx==2.2
pip uninstall numpy && pip install numpy==1.19.5
pip uninstall tensorboard && pip install tensorboard==2.14.0
pip uninstall pandas && pip install pandas==1.4.4
pip uninstall yapf && pip install yapf==0.40.1
pip uninstall setuptools && pip install setuptools==59.5.0
```

```bash
pip install protobuf==4.25.3
pip install markdown==3.5.2
pip install absl-py==2.1.0
pip install grpcio==1.62.0
pip install einops==0.7.0
```


### DETR3D
**Option 1** : Follow the installation instructions of [DETR3D](https://github.com/WangYueFt/detr3d). If some versions of packages are not provided, please refer to our [requirements](./requirements_detr3d.txt) for DETR3D.

**Option 2** : 
```bash
conda create --name detr3d python=3.8
conda activate detr3d
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch
```

```bash
pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.1/index.html
pip install mmdet==2.24.1
pip install mmsegmentation==0.20.2
cd UAV3D/perception/detr3d/mmdetection3d
python3 setup.py develop
```

```bash
pip install scikit-image==0.19.3
pip install numba==0.48.0
pip install  nuscenes-devkit==1.1.10
pip install  lyft-dataset-sdk==0.0.8
pip install  plyfile==1.0.3
pip install networkx==2.2
pip install numpy==1.19.5
pip install tensorboard==2.14.0
pip install pandas==1.4.4
pip install yapf==0.40.1
pip install setuptools==59.5.0
```

### CenterPoint
**Option 1** : For some versions of packages, please refer to our [requirements](./requirements_centerpoint.txt) for CenterPoint. **Note**: We have added some python scripts to the __CenterPoint__ repo. Please use this [version](./tracking/CenterPoint) for __UAV3D__. 

```bash
conda create --name centerpoint python=3.8
conda activate centerpoint
```

```bash
git clone https://github.com/huiyegit/UAV3D.git
cd UAV3D/tracking/CenterPoint
pip install -r requirements
```


## Docker environment
**Option 1** : Follow the installation instructions for Docker in [BEVFusion](https://github.com/mit-han-lab/bevfusion). Then follow the instructions __above__ to install the Conda enviroment in Docker. 







