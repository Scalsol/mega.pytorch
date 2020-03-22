import os
import pickle

import torch
import torch.utils.data

from PIL import Image
import cv2
import sys
import numpy as np

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

from mega_core.structures.bounding_box import BoxList
from mega_core.utils.comm import is_main_process


class VIDDataset(torch.utils.data.Dataset):
    classes = ['__background__',  # always index 0
                'airplane', 'antelope', 'bear', 'bicycle',
                'bird', 'bus', 'car', 'cattle',
                'dog', 'domestic_cat', 'elephant', 'fox',
                'giant_panda', 'hamster', 'horse', 'lion',
                'lizard', 'monkey', 'motorcycle', 'rabbit',
                'red_panda', 'sheep', 'snake', 'squirrel',
                'tiger', 'train', 'turtle', 'watercraft',
                'whale', 'zebra']
    classes_map = ['__background__',  # always index 0
                    'n02691156', 'n02419796', 'n02131653', 'n02834778',
                    'n01503061', 'n02924116', 'n02958343', 'n02402425',
                    'n02084071', 'n02121808', 'n02503517', 'n02118333',
                    'n02510455', 'n02342885', 'n02374451', 'n02129165',
                    'n01674464', 'n02484322', 'n03790512', 'n02324045',
                    'n02509815', 'n02411705', 'n01726692', 'n02355227',
                    'n02129604', 'n04468005', 'n01662784', 'n04530566',
                    'n02062744', 'n02391049']

    def __init__(self, image_set, data_dir, img_dir, anno_path, img_index, transforms, is_train=True):
        self.det_vid = image_set.split("_")[0]
        self.image_set = image_set
        self.transforms = transforms

        self.data_dir = data_dir
        self.img_dir = img_dir
        self.anno_path = anno_path
        self.img_index = img_index

        self.is_train = is_train

        self._img_dir = os.path.join(self.img_dir, "%s.JPEG")
        self._anno_path = os.path.join(self.anno_path, "%s.xml")

        with open(self.img_index) as f:
            lines = [x.strip().split(" ") for x in f.readlines()]
        if len(lines[0]) == 2:
            self.image_set_index = [x[0] for x in lines]
            self.frame_id = [int(x[1]) for x in lines]
        else:
            self.image_set_index = ["%s/%06d" % (x[0], int(x[2])) for x in lines]
            self.pattern = [x[0] + "/%06d" for x in lines]
            self.frame_id = [int(x[1]) for x in lines]
            self.frame_seg_id = [int(x[2]) for x in lines]
            self.frame_seg_len = [int(x[3]) for x in lines]

        if self.is_train:
            keep = self.filter_annotation()

            if len(lines[0]) == 2:
                self.image_set_index = [self.image_set_index[idx] for idx in range(len(keep)) if keep[idx]]
                self.frame_id = [self.frame_id[idx] for idx in range(len(keep)) if keep[idx]]
            else:
                self.image_set_index = [self.image_set_index[idx] for idx in range(len(keep)) if keep[idx]]
                self.pattern = [self.pattern[idx] for idx in range(len(keep)) if keep[idx]]
                self.frame_id = [self.frame_id[idx] for idx in range(len(keep)) if keep[idx]]
                self.frame_seg_id = [self.frame_seg_id[idx] for idx in range(len(keep)) if keep[idx]]
                self.frame_seg_len = [self.frame_seg_len[idx] for idx in range(len(keep)) if keep[idx]]

        self.classes_to_ind = dict(zip(self.classes_map, range(len(self.classes_map))))
        self.categories = dict(zip(range(len(self.classes)), self.classes))

        self.annos = self.load_annos(os.path.join(self.cache_dir, self.image_set + "_anno.pkl"))

    def __getitem__(self, idx):
        if self.is_train:
            return self._get_train(idx)
        else:
            return self._get_test(idx)

    def _get_train(self, idx):
        filename = self.image_set_index[idx]
        img = Image.open(self._img_dir % filename).convert("RGB")

        target = self.get_groundtruth(idx)
        target = target.clip_to_image(remove_empty=True)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, idx

    def _get_test(self, idx):
        return self._get_train(idx)

    def __len__(self):
        return len(self.image_set_index)

    def filter_annotation(self):
        cache_file =os.path.join(self.cache_dir, self.image_set + "_keep.pkl")

        if os.path.exists(cache_file):
            with open(cache_file, "rb") as fid:
                keep = pickle.load(fid)
            if is_main_process():
                print("{}'s keep information loaded from {}".format(self.det_vid, cache_file))
            return keep

        keep = np.zeros((len(self)), dtype=np.bool)
        for idx in range(len(self)):
            if idx % 10000 == 0:
                print("Had filtered {} images".format(idx))

            filename = self.image_set_index[idx]

            tree = ET.parse(self._anno_path % filename).getroot()
            objs = tree.findall("object")
            keep[idx] = False if len(objs) == 0 else True
        print("Had filtered {} images".format(len(self)))

        if is_main_process():
            with open(cache_file, "wb") as fid:
                pickle.dump(keep, fid)
            print("Saving {}'s keep information into {}".format(self.det_vid, cache_file))

        return keep

    def _preprocess_annotation(self, target):
        boxes = []
        gt_classes = []

        size = target.find("size")
        im_info = tuple(map(int, (size.find("height").text, size.find("width").text)))

        objs = target.findall("object")
        for obj in objs:
            if not obj.find("name").text in self.classes_to_ind:
                continue

            bbox =obj.find("bndbox")
            box = [
                np.maximum(float(bbox.find("xmin").text), 0),
                np.maximum(float(bbox.find("ymin").text), 0),
                np.minimum(float(bbox.find("xmax").text), im_info[1] - 1),
                np.minimum(float(bbox.find("ymax").text), im_info[0] - 1)
            ]
            boxes.append(box)
            gt_classes.append(self.classes_to_ind[obj.find("name").text.lower().strip()])

        res = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(gt_classes),
            "im_info": im_info,
        }
        return res

    def load_annos(self, cache_file):
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as fid:
                annos = pickle.load(fid)
            if is_main_process():
                print("{}'s annotation information loaded from {}".format(self.det_vid, cache_file))
        else:
            annos = []
            for idx in range(len(self)):
                if idx % 10000 == 0:
                    print("Had processed {} images".format(idx))

                filename = self.image_set_index[idx]

                tree = ET.parse(self._anno_path % filename).getroot()
                anno = self._preprocess_annotation(tree)
                annos.append(anno)
            print("Had processed {} images".format(len(self)))

            if is_main_process():
                with open(cache_file, "wb") as fid:
                    pickle.dump(annos, fid)
                print("Saving {}'s annotation information into {}".format(self.det_vid, cache_file))

        return annos

    def get_img_info(self, idx):
        im_info = self.annos[idx]["im_info"]
        return {"height": im_info[0], "width": im_info[1]}

    @property
    def cache_dir(self):
        """
        make a directory to store all caches
        :return: cache path
        """
        cache_dir = os.path.join(self.data_dir, 'cache')
        if not os.path.exists(cache_dir):
            os.mkdir(cache_dir)
        return cache_dir

    def get_visualization(self, idx):
        filename = self.image_set_index[idx]

        img = cv2.imread(self._img_dir % filename)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        target = self.get_groundtruth(idx)
        target = target.clip_to_image(remove_empty=True)

        return img, target, filename

    def get_groundtruth(self, idx):
        anno = self.annos[idx]

        height, width = anno["im_info"]
        target = BoxList(anno["boxes"].reshape(-1, 4), (width, height), mode="xyxy")
        target.add_field("labels", anno["labels"])

        return target

    @staticmethod
    def map_class_id_to_class_name(class_id):
        return VIDDataset.classes[class_id]
