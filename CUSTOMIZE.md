# Customize

This file will give a brief introduction for:
 - How to prepare your own datasets for training with provided methods.
 - How to implement your own fascinating method.
 
## Prepare your own dataset

We would first give some notations.

- Prefix: a string that is used to identify your dataset. Suppose it is `vid_custom` here.
- Split: We need to prepare a `train` and `val` split for training and evaluating our method. When combined with the prefix, they become `vid_custom_train` and `vid_custom_val`.

Then we will go through the whole pipeline.

### Structure your dataset

We suggest you to organize your data structure as the ImageNet VID dataset. It should look like:

```
datasets
├── vid_custom
|   |── train
|   |   |── video_snippet_1
|   |   |   |── 000000.JPEG
|   |   |   |── 000001.JPEG
|   |   |   |── 000002.JPEG
|   |   |   ...
|   |   |── video_snippet_2
|   |   |   |── 000000.JPEG
|   |   |   |── 000001.JPEG
|   |   |   |── 000002.JPEG
|   |   |   ...
|   |   ...
|   |── val
|   |   |── video_snippet_1
|   |   |   |── 000000.JPEG
|   |   |   |── 000001.JPEG
|   |   |   |── 000002.JPEG
|   |   |   ...
|   |   |── video_snippet_2
|   |   |   |── 000000.JPEG
|   |   |   |── 000001.JPEG
|   |   |   |── 000002.JPEG
|   |   |   ...
|   |   ...
|   |── annotation
|   |   |   |── train
|   |   |   |── val
``` 

Following this structure, your could directly use all provided methods with only minor modification on the dataloader, which will be introduced later.

### Prepare your txt file

After sturcturing the dataset, then we need to give a indexing file to let the dataloader know which set of frames are used to train the model. We suggest to prepare this file as [VID_train_15frames.txt](datasets/ILSVRC2015/ImageSets/VID_train_15frames.txt). This txt file should have four strings in a single line: `video folder`, `no meaning`(just ignore it), `frame number`, `video length`. That should be enough.

### Prepare your dataloader

Once you have prepared your dataset, we should assign a dataloader to load images and annotations. Notice differnet method uses different dataloader. Take single frame baseline as an example, we use [baseloader](mega_core/data/datasets/vid.py) to load the data. To make it compatible with your dataset, you should modify:

- You should modify attribute `classes` to the categories in your dataset.
- Modify the `load_annos()` and `_preprocess_annotation()` method to make it compatible with your annotation style. Make sure the returned field is the same as the baseloader.

Then the preparation is done. Very easy, right? If your want to use other methold, you could directly inherit those dataloaders from your newly created baseloader. No additional changes are needed.

### Register your dataset

Once you have done the above steps, the dataset and the dataloader needs to be added in a couple of places:

- [`mega_core/data/datasets/__init__.py`](mega_core/data/datasets/__init__.py): add the dataloader to `__all__`
- [`mega_core/config/paths_catalog.py`](mega_core/config/paths_catalog.py): add `vid_custom_train` and `vid_custom_val` as a dictionary name with field `img_dir`, `anno_path` and `img_index` in `DatasetCatalog.DATASETS`. And corresponding `if` clause in `DatasetCatalog.get()`.

### Modify the base config

Modify the `configs/BASE_RCNN_Xgpu.yaml` to make it compatible with the statistics of your dataset, e.g., the `NUM_CLASSES`.

## Implement your own method 

First give your method a fancy name, so suppose your method's name is `fancy` here.

### Prepare a new dataloader

Create a new dataloader under folder `data/dataset`, it should be inherited from the `VIDEODataset` class. You only need to make a minor modification on `__init__()` method (see [`vid_fgfa.py`](mega_core/data/datasets/vid_fgfa.py) for example) and implement `_get_train()` and `_get_test()` method.

As video object detection methods usually require some reference frames to assist the detection on current frame. We recommend that the current frame should be stored in `images["cur"]` and all reference frames be stored in `images["ref"]` as a list. This will make the following batch collating procedure easier. But it all depends on you. see [`vid_fgfa.py`](mega_core/data/datasets/vid_fgfa.py) for a example. 

Once you have created your dataloader, it needs to be added in a couple of places:
- [`mega_core/data/collate_batch.py`](mega_core/data/collate_batch.py): add your method name `fancy` in the `if` clause in `BatchCollator.__call__()`. And modify the processing step to make it compatible with your dataloader behavior.
- [`mega_core/data/datasets/__init__.py`](mega_core/data/datasets/__init__.py): add the dataloader to `__all__`.
- [`mega_core/config/paths_catalog.py`](mega_core/config/paths_catalog.py): add corresponding `if` clause in `DatasetCatalog.get()` to access your method.


### Prepare your model

Create your model under directory `mega_core/modeling/detector` and register it in [`mega_core/modeling/detector/detectors.py`](mega_core/modeling/detector/detectors.py). Take [`mega_core/modeling/detector/generalized_rcnn_mega.py`](mega_core/modeling/detector/generalized_rcnn_mega.py) and the corresponding config files as reference if you use new submodules in your model. 

### Register your method

- [`mega_core/engine/trainer.py`](mega_core/engine/trainer.py): Line 87
- [`mega_core/engine/inference.py`](mega_core/engine/inference.py): Line 31
- [`mega_core/modeling/rpn/rpn.py`](mega_core/modeling/rpn/rpn.py): Line 254
- [`mega_core/data/build.py`](mega_core/data/build.py): Line 66

### Prepare config files

- The field `MODEL.VID.METHOD` should be specified as `fancy`.
- The field `MODEL.META_ARCHITECTURE` should be your model name.
- If you feel confused, take a look at other configs.

If you still feel confused with some steps above or some instructions are wrong, please contact me to fix it or make it more clear.
