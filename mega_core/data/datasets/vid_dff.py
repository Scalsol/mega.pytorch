from PIL import Image
import sys
import numpy as np

from .vid import VIDDataset
from mega_core.config import cfg

class VIDDFFDataset(VIDDataset):
    def __init__(self, image_set, data_dir, img_dir, anno_path, img_index, transforms, is_train=True):
        super(VIDDFFDataset, self).__init__(image_set, data_dir, img_dir, anno_path, img_index, transforms, is_train=is_train)
        if not self.is_train:
            self.start_index = []
            for id, image_index in enumerate(self.image_set_index):
                frame_id = int(image_index.split("/")[-1])
                if frame_id == 0:
                    self.start_index.append(id)

    def _get_train(self, idx):
        filename = self.image_set_index[idx]
        img = Image.open(self._img_dir % filename).convert("RGB")

        # if a video dataset
        img_refs = []
        if hasattr(self, "pattern"):
            offsets = np.random.choice(cfg.MODEL.VID.DFF.MAX_OFFSET - cfg.MODEL.VID.DFF.MIN_OFFSET + 1, 1, replace=False) + cfg.MODEL.VID.DFF.MIN_OFFSET
            for i in range(len(offsets)):
                ref_id = min(max(self.frame_seg_id[idx] + offsets[i], 0), self.frame_seg_len[idx] - 1)
                ref_filename = self.pattern[idx] % ref_id
                img_ref = Image.open(self._img_dir % ref_filename).convert("RGB")
                img_refs.append(img_ref)
        else:
            img_refs.append(img.copy())

        target = self.get_groundtruth(idx)
        target = target.clip_to_image(remove_empty=True)

        if self.transforms is not None:
            img, target = self.transforms(img, target)
            for i in range(len(img_refs)):
                img_refs[i], _ = self.transforms(img_refs[i], None)

        images = {}
        images["cur"] = img
        images["ref"] = img_refs

        return images, target, idx

    def _get_test(self, idx):
        filename = self.image_set_index[idx]
        img = Image.open(self._img_dir % filename).convert("RGB")

        is_key_frame = False
        frame_id = int(filename.split("/")[-1])
        if frame_id % 10 == 0:
            is_key_frame = True

        target = self.get_groundtruth(idx)
        target = target.clip_to_image(remove_empty=True)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        images = {}
        images["cur"] = img
        images["is_key_frame"] = is_key_frame

        return images, target, idx
