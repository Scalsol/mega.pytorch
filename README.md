# MEGA for Video Object Detection

[![License](https://img.shields.io/badge/license-BSD-blue.svg)](LICENSE)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/memory-enhanced-global-local-aggregation-for/video-object-detection-on-imagenet-vid)](https://paperswithcode.com/sota/video-object-detection-on-imagenet-vid?p=memory-enhanced-global-local-aggregation-for)

By [Yihong Chen](https://scalsol.github.io), [Yue Cao](http://yue-cao.me), [Han Hu](https://ancientmooner.github.io/), [Liwei Wang](http://www.liweiwang-pku.com/).

This repo is an official implementation of ["Memory Enhanced Global-Local Aggregation for Video Object Detection"](https://arxiv.org/abs/2003.12063), accepted by CVPR 2020. This repository contains a PyTorch implementation of our approach MEGA based on [maskrcnn_benchmark](https://github.com/facebookresearch/maskrcnn-benchmark), as well as some training scripts to reproduce the results on ImageNet VID reported in our paper. 

Besides, this repository also implements several other algorithms like [FGFA](http://openaccess.thecvf.com/content_iccv_2017/html/Zhu_Flow-Guided_Feature_Aggregation_ICCV_2017_paper.html) and [RDN](https://arxiv.org/abs/1908.09511). Any new methods are welcomed. Hoping for your pull request! We hope this repository would help further research in the field of video object detection and beyond. :)

## Citing MEGA
Please cite our paper in your publications if it helps your research:
```
@inproceedings{chen20mega,
    Author = {Chen, Yihong and Cao, Yue and Hu, Han and Wang, Liwei},
    Title = {Memory Enhanced Global-Local Aggregation for Video Object Detection},
    Conference = {CVPR},
    Year = {2020}
}
```

## Updates

- Add motion-IoU specific AP evaluation code. Only available for ImageNet VID dataset. (19/06/2020)
- Demo for visualization added (Support image folder and video). (17/06/2020)
- Results of ResNet-50 backbone added. (13/04/2020)
- Code and pretrained weights for [Deep Feature Flow](https://arxiv.org/abs/1611.07715) released. (30/03/2020)

## Main Results

Pretrained models are now available at [Baidu](https://pan.baidu.com/s/1qjIAD3ohaJO8EF1mZ4nLEg) (code: neck) and Google Drive.

Model | Backbone | AP50 | AP (fast) | AP (med) | AP (slow) | Link
:---: | :---: | :---: | :---: | :---: | :---: |:---:
single frame baseline | ResNet-101 | 76.7 | 52.3 | 74.1 | 84.9 | [Google](https://drive.google.com/file/d/1W17f9GC60rHU47lUeOEfU--Ra-LTw3Tq/view?usp=sharing)
DFF | ResNet-101 | 75.0 | 48.3 | 73.5 | 84.5 | [Google](https://drive.google.com/file/d/1Dn_RQRlA7z2XkRRS4XERUW_UH9jlNvMo/view?usp=sharing)
FGFA | ResNet-101 | 78.0 | 55.3 | 76.9 | 85.6 | [Google](https://drive.google.com/file/d/1yVgy7_ff1xVD1SooqbcK-OzKMgPpUcg4/view?usp=sharing)
RDN-base | ResNet-101 | 81.1 | 60.2 | 79.4 | 87.7 | [Google](https://drive.google.com/file/d/1jM5LqlVtCGjKH-MocTCjzFIVjqCyng8M/view?usp=sharing)
RDN | ResNet-101 | 81.7 | 59.5 | 80.0 | 89.0| [Google](https://drive.google.com/file/d/1FgoOwj-GFAMVn2hkSFKnxn5fKWPSxlUF/view?usp=sharing)
**MEGA** | ResNet-101 | 82.9 | 62.7| 81.6 | 89.4 | [Google](https://drive.google.com/file/d/1ZnAdFafF1vW9Lnpw-RPF1AD_csw61lBY/view?usp=sharing)

Model | Backbone | AP50 | AP (fast) | AP (med) | AP (slow) | Link
:---: | :---: | :---: | :---: | :---: | :---: |:---:
single frame baseline | ResNet-50 | 71.8 | 47.2 | 69.2 | 80.6| [Google](https://drive.google.com/file/d/1i39MwpP46x61eHLkRXMzcKhpeKZhkgA6/view?usp=sharing)
DFF | ResNet-50 | 70.4 | 43.6 | 68.9 | 80.8 | [Google](https://drive.google.com/file/d/1wl9Sheg46ecJOWzl1Uy4BWaCDRtSt51_/view?usp=sharing)
FGFA | ResNet-50 | 74.3 | 50.6 | 72.3 | 84.0|  [Google](https://drive.google.com/file/d/1nJ6CbUG_wW_gvMs193b7f0c1QLnXqAzO/view?usp=sharing)
RDN-base | ResNet-50 | 76.7 | 53.8 | 74.8 | 85.4 | [Google](https://drive.google.com/file/d/10k70lzSrxXiLWYx8tmX3RNuOQ2x1X0k8/view?usp=sharing)
**MEGA** | ResNet-50 | 77.3 | 56.5 | 75.7 | 85.2 | [Google](https://drive.google.com/file/d/1EZzpBuCfI75bsd_gxK1495tXlh0K_34H/view?usp=sharing)

**Note**: The performance of ResNet-50 backbone are not so stable.

**Note**: The motion-IoU specific AP evaluation code is a bit different from the original implementation in FGFA. I think the original implementation is really weird so I modify it. So the results may not be directly comparable with the results provided in FGFA and other methods that use MXNet version evaluation code. But we could tell which method is relatively better under the same evaluation protocol. 

## Installation

Please follow [INSTALL.md](INSTALL.md) for installation instructions.

## Data preparation

Please download ILSVRC2015 DET and ILSVRC2015 VID dataset from [here](http://image-net.org/challenges/LSVRC/2015/2015-downloads). After that, we recommend to symlink the path to the datasets to `datasets/`. And the path structure should be as follows:

    ./datasets/ILSVRC2015/
    ./datasets/ILSVRC2015/Annotations/DET
    ./datasets/ILSVRC2015/Annotations/VID
    ./datasets/ILSVRC2015/Data/DET
    ./datasets/ILSVRC2015/Data/VID
    ./datasets/ILSVRC2015/ImageSets
    
**Note**: We have already provided a list of all images we use to train and test our model as txt files under directory `datasets/ILSVRC2015/ImageSets`. You do not need to change them.

## Usage

**Note**: Cache files will be created at the first time you run this project, this may take some time! Don't worry!

**Note**: Currently, one GPU could only hold 1 image. Do not put 2 or more images on 1 GPU!

**Note** We provide template files named `BASE_RCNN_{}gpus.yaml` which would automatically change the batch size and other relevant settings. This behavior is similar to detectron2. If you want to train model with different number of gpus, please change it by yourself :) But assure **1 GPU only holds 1 image**! That is to say, you should always keep `SOLVER.IMS_PER_BATCH` and `TEST.IMS_PER_BATCH` equal to the number of GPUs you use.

### Inference

The inference command line for testing on the validation dataset:

    python -m torch.distributed.launch \
        --nproc_per_node 4 \
        tools/test_net.py \
        --config-file configs/MEGA/vid_R_101_C4_MEGA_1x.yaml \
        --motion-specific \
        MODEL.WEIGHT MEGA_R_101.pth 
        
Please note that:
1) If your model's name is different, please replace `MEGA_R_101.pth` with your own.
2) If you want to evaluate a different model, please change `--config-file` to its config file and `MODEL.WEIGHT` to its weights file.
3) If you do not want to evaluate motion-IoU specific AP, simply deleting `--motion-specific`.
4) Testing is time-consuming, so be patient!
5) As testing on above 170000+ frames is toooo time-consuming, so we enable directly testing on generated bounding boxes, which is automatically saved in a file named `predictions.pth` on your training directory. That means you do not need to run the evaluation from the very start every time. You could access this by running:
```
    python tools/test_prediction.py \
        --config-file configs/MEGA/vid_R_101_C4_MEGA_1x.yaml \
        --prediction [YOUR predictions.pth generated by MEGA]
        --motion-specific
```

### Training

The following command line will train MEGA_R_101_FPN_1x on 4 GPUs with Synchronous Stochastic Gradient Descent (SGD):

    python -m torch.distributed.launch \
        --nproc_per_node=4 \
        tools/train_net.py \
        --master_port=$((RANDOM + 10000)) \
        --config-file configs/MEGA/vid_R_101_C4_MEGA_1x.yaml \
        --motion-specific \
        OUTPUT_DIR training_dir/MEGA_R_101_1x
        
Please note that:
1) The models will be saved into `OUTPUT_DIR`.
2) If you want to train MEGA and other methods with other backbones, please change `--config-file`.
3) For training FGFA and DFF, we need pretrained weight of FlowNet. We provide the converted version [here](https://drive.google.com/file/d/1gib7XtS1fSYDTM9RnUJ72a3vREV_6SJH/view?usp=sharing). After downloaded, it should be placed at `models/`. See `config/defaults.py` and the code for further details.
4) For training RDN, we adopt the same two-stage training strategy as described in its original paper. The first phase should be run with config file `configs/RDN/vid_R_101_C4_RDN_base_1x.yaml`. For the second phase, `MODEL.WEIGHT` should be set to the filename of the final model of the first stage training. Or you could rename the model's filename to `RDN_base_R_101.pth` and put it under `models/` and directly train the second phase with config file `configs/RDN/vid_R_101_C4_RDN_1x.yaml`.
5) If you do not want to evaluate motion-IoU specific AP at the end of training, simply deleting `--motion-specific`.

### Demo Usage
Please follow [demo/README.md](demo/README.md) to see how to visualize your own images or video.

### Customize

If you want to use these methods on your own dataset or implement your new method. Please follow [CUSTOMIZE.md](CUSTOMIZE.md).

## Contributing to the project
Any pull requests or issues are welcomed.
