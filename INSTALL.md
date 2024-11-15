## Installation

### Requirements:
- python >= 3.8
- PyTorch == 2.5.1
- torchvision from master
- cocoapi
- yacs
- matplotlib
- GCC >= 4.9
- OpenCV
- CUDA >= 12.4


### Option 1: Step-by-step installation

```bash
# first, make sure that your conda is setup properly with the right environment
# for that, check that `which conda`, `which pip` and `which python` points to the
# right path. From a clean conda env, this is what you need to do

conda create --name MEGA -y python=3.12
source activate MEGA

# this installs the right pip and dependencies for the fresh python
conda install ipython pip

# mega and coco api dependencies
pip install build wheel installer ninja yacs cython matplotlib tqdm opencv-python scipy numpy

# follow PyTorch installation in https://pytorch.org/get-started/locally/
pip install torch torchvision torchaudio

export INSTALL_DIR=$PWD

# install pycocotools
cd $INSTALL_DIR
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python -m build --wheel --no-isolation
python -m installer dist/*.whl

# install cityscapesScripts
cd $INSTALL_DIR
git clone https://github.com/mcordts/cityscapesScripts.git
cd cityscapesScripts/
python -m build --wheel --no-isolation
python -m installer dist/*.whl

# install apex
cd $INSTALL_DIR
git clone https://github.com/NVIDIA/apex.git
cd apex
python -m build --wheel --no-isolation
python -m installer dist/*.whl

# install PyTorch Detection
cd $INSTALL_DIR
git clone https://github.com/Scalsol/mega.pytorch.git
cd mega.pytorch
python -m build --wheel --no-isolation
python -m installer dist/*.whl

unset INSTALL_DIR

# or if you are on macOS
# MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ python setup.py build develop
```
