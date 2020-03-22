from PIL import Image
import sys
import numpy as np

from .vid import VIDDataset
from mega_core.config import cfg

class VIDRDNDataset(VIDDataset):
    def __init__(self, image_set, data_dir, img_dir, anno_path, img_index, transforms, is_train=True):
        super(VIDRDNDataset, self).__init__(image_set, data_dir, img_dir, anno_path, img_index, transforms, is_train=is_train)
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
            offsets = np.random.choice(cfg.MODEL.VID.RDN.MAX_OFFSET - cfg.MODEL.VID.RDN.MIN_OFFSET + 1, cfg.MODEL.VID.RDN.REF_NUM, replace=False) + cfg.MODEL.VID.RDN.MIN_OFFSET
            for i in range(len(offsets)):
                ref_id = min(max(self.frame_seg_id[idx] + offsets[i], 0), self.frame_seg_len[idx] - 1)
                ref_filename = self.pattern[idx] % ref_id
                img_ref = Image.open(self._img_dir % ref_filename).convert("RGB")
                img_refs.append(img_ref)
        else:
            for i in range(cfg.MODEL.VID.RDN.REF_NUM):
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

        # give the current frame a category. 0 for start, 1 for normal
        frame_id = int(filename.split("/")[-1])
        frame_category = 0
        if frame_id != 0:
            frame_category = 1

        img_refs = []
        # reading other images of the queue (not necessary to be the last one, but last one here)
        ref_id = min(self.frame_seg_len[idx] - 1, frame_id + cfg.MODEL.VID.RDN.MAX_OFFSET)
        ref_filename = self.pattern[idx] % ref_id
        img_ref = Image.open(self._img_dir % ref_filename).convert("RGB")
        img_refs.append(img_ref)

        target = self.get_groundtruth(idx)
        target = target.clip_to_image(remove_empty=True)

        if self.transforms is not None:
            img, target = self.transforms(img, target)
            for i in range(len(img_refs)):
                img_refs[i], _ = self.transforms(img_refs[i], None)

        images = {}
        images["cur"] = img
        images["ref"] = img_refs
        images["frame_category"] = frame_category
        images["seg_len"] = self.frame_seg_len[idx]
        images["pattern"] = self.pattern[idx]
        images["img_dir"] = self._img_dir
        images["transforms"] = self.transforms

        return images, target, idx
