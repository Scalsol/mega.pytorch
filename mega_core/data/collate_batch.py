# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from mega_core.structures.image_list import to_image_list


class BatchCollator(object):
    """
    From a list of samples from the dataset,
    returns the batched images and targets.
    This should be passed to the DataLoader
    """

    def __init__(self, size_divisible=0, method="base", is_train=True):
        self.size_divisible = size_divisible
        self.method = method
        self.is_train = is_train

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        if self.method in ("base", ):
            images = to_image_list(transposed_batch[0], self.size_divisible)
        elif self.method in ("rdn", "mega", "fgfa", "dff"):
            assert len(transposed_batch[0]) == 1, "Currently 1 gpu could only hold 1 image. Please modify SOLVER.IMS_PER_BATCH and TEST.IMS_PER_BATCH to ensure this."
            images = {}
            for key in transposed_batch[0][0].keys():
                if key == "cur":
                    images["cur"] = to_image_list((transposed_batch[0][0]["cur"], ), self.size_divisible)
                if key not in ("ref", "ref_l", "ref_m", "ref_g"):
                    images[key] = transposed_batch[0][0][key]
                else:
                    if transposed_batch[0][0][key]:
                        images[key] = [to_image_list((img,), self.size_divisible) for img in transposed_batch[0][0][key]]
                    else:
                        images[key] = []
        else:
            raise NotImplementedError("method {} not supported yet.".format(self.method))

        targets = transposed_batch[1]
        img_ids = transposed_batch[2]
        return images, targets, img_ids


class BBoxAugCollator(object):
    """
    From a list of samples from the dataset,
    returns the images and targets.
    Images should be converted to batched images in `im_detect_bbox_aug`
    """

    def __call__(self, batch):
        return list(zip(*batch))
