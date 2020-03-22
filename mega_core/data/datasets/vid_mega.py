from PIL import Image
import sys
import numpy as np

from .vid import VIDDataset
from mega_core.config import cfg

class VIDMEGADataset(VIDDataset):
    def __init__(self, image_set, data_dir, img_dir, anno_path, img_index, transforms, is_train=True):
        super(VIDMEGADataset, self).__init__(image_set, data_dir, img_dir, anno_path, img_index, transforms, is_train=is_train)
        if not self.is_train:
            self.start_index = []
            self.start_id = []
            if cfg.MODEL.VID.MEGA.GLOBAL.ENABLE:
                self.shuffled_index = {}
            for id, image_index in enumerate(self.image_set_index):
                frame_id = int(image_index.split("/")[-1])
                if frame_id == 0:
                    self.start_index.append(id)
                    if cfg.MODEL.VID.MEGA.GLOBAL.ENABLE:
                        shuffled_index = np.arange(self.frame_seg_len[id])
                        if cfg.MODEL.VID.MEGA.GLOBAL.SHUFFLE:
                            np.random.shuffle(shuffled_index)
                        self.shuffled_index[str(id)] = shuffled_index

                    self.start_id.append(id)
                else:
                    self.start_id.append(self.start_index[-1])

    def _get_train(self, idx):
        filename = self.image_set_index[idx]
        img = Image.open(self._img_dir % filename).convert("RGB")

        # if a video dataset
        img_refs_l = []
        img_refs_m = []
        img_refs_g = []
        if hasattr(self, "pattern"):
            offsets = np.random.choice(cfg.MODEL.VID.MEGA.MAX_OFFSET - cfg.MODEL.VID.MEGA.MIN_OFFSET + 1,
                                       cfg.MODEL.VID.MEGA.REF_NUM_LOCAL, replace=False) + cfg.MODEL.VID.MEGA.MIN_OFFSET
            for i in range(len(offsets)):
                ref_id = min(max(self.frame_seg_id[idx] + offsets[i], 0), self.frame_seg_len[idx] - 1)
                ref_filename = self.pattern[idx] % ref_id
                img_ref = Image.open(self._img_dir % ref_filename).convert("RGB")
                img_refs_l.append(img_ref)

            # memory frames
            if cfg.MODEL.VID.MEGA.MEMORY.ENABLE:
                ref_id_center = max(self.frame_seg_id[idx] - cfg.MODEL.VID.MEGA.ALL_FRAME_INTERVAL, 0)
                offsets = np.random.choice(cfg.MODEL.VID.MEGA.MAX_OFFSET - cfg.MODEL.VID.MEGA.MIN_OFFSET + 1,
                                           cfg.MODEL.VID.MEGA.REF_NUM_MEM, replace=False) + cfg.MODEL.VID.MEGA.MIN_OFFSET
                for i in range(len(offsets)):
                    ref_id = min(max(ref_id_center + offsets[i], 0), self.frame_seg_len[idx] - 1)
                    ref_filename = self.pattern[idx] % ref_id
                    img_ref = Image.open(self._img_dir % ref_filename).convert("RGB")
                    img_refs_m.append(img_ref)

            # global frames
            if cfg.MODEL.VID.MEGA.GLOBAL.ENABLE:
                ref_ids = np.random.choice(self.frame_seg_len[idx], cfg.MODEL.VID.MEGA.REF_NUM_GLOBAL, replace=False)
                for ref_id in ref_ids:
                    ref_filename = self.pattern[idx] % ref_id
                    img_ref = Image.open(self._img_dir % ref_filename).convert("RGB")
                    img_refs_g.append(img_ref)
        else:
            for i in range(cfg.MODEL.VID.MEGA.REF_NUM_LOCAL):
                img_refs_l.append(img.copy())
            if cfg.MODEL.VID.MEGA.MEMORY.ENABLE:
                for i in range(cfg.MODEL.VID.MEGA.REF_NUM_MEM):
                    img_refs_m.append(img.copy())
            if cfg.MODEL.VID.MEGA.GLOBAL.ENABLE:
                for i in range(cfg.MODEL.VID.MEGA.REF_NUM_GLOBAL):
                    img_refs_g.append(img.copy())

        target = self.get_groundtruth(idx)
        target = target.clip_to_image(remove_empty=True)

        if self.transforms is not None:
            img, target = self.transforms(img, target)
            for i in range(len(img_refs_l)):
                img_refs_l[i], _ = self.transforms(img_refs_l[i], None)
            for i in range(len(img_refs_m)):
                img_refs_m[i], _ = self.transforms(img_refs_m[i], None)
            for i in range(len(img_refs_g)):
                img_refs_g[i], _ = self.transforms(img_refs_g[i], None)

        images = {}
        images["cur"] = img
        images["ref_l"] = img_refs_l
        images["ref_m"] = img_refs_m
        images["ref_g"] = img_refs_g

        return images, target, idx

    def _get_test(self, idx):
        filename = self.image_set_index[idx]
        img = Image.open(self._img_dir % filename).convert("RGB")

        # give the current frame a category. 0 for start, 1 for normal
        frame_id = int(filename.split("/")[-1])
        frame_category = 0
        if frame_id != 0:
            frame_category = 1

        img_refs_l = []
        # reading other images of the queue (not necessary to be the last one, but last one here)
        ref_id = min(self.frame_seg_len[idx] - 1, frame_id + cfg.MODEL.VID.MEGA.MAX_OFFSET)
        ref_filename = self.pattern[idx] % ref_id
        img_ref = Image.open(self._img_dir % ref_filename).convert("RGB")
        img_refs_l.append(img_ref)

        img_refs_g = []
        if cfg.MODEL.VID.MEGA.GLOBAL.ENABLE:
            size = cfg.MODEL.VID.MEGA.GLOBAL.SIZE if frame_id == 0 else 1
            shuffled_index = self.shuffled_index[str(self.start_id[idx])]
            for id in range(size):
                filename = self.pattern[idx] % shuffled_index[
                    (idx - self.start_id[idx] + cfg.MODEL.VID.MEGA.GLOBAL.SIZE - id - 1) % self.frame_seg_len[idx]]
                img = Image.open(self._img_dir % filename).convert("RGB")
                img_refs_g.append(img)

        target = self.get_groundtruth(idx)
        target = target.clip_to_image(remove_empty=True)

        if self.transforms is not None:
            img, target = self.transforms(img, target)
            for i in range(len(img_refs_l)):
                img_refs_l[i], _ = self.transforms(img_refs_l[i], None)
            for i in range(len(img_refs_g)):
                img_refs_g[i], _ = self.transforms(img_refs_g[i], None)

        images = {}
        images["cur"] = img
        images["ref_l"] = img_refs_l
        images["ref_g"] = img_refs_g
        images["frame_category"] = frame_category
        images["seg_len"] = self.frame_seg_len[idx]
        images["pattern"] = self.pattern[idx]
        images["img_dir"] = self._img_dir
        images["transforms"] = self.transforms

        return images, target, idx
