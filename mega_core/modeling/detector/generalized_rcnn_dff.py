# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""
from PIL import Image
from collections import deque

import torch
from torch import nn
import torch.nn.functional as F

from mega_core.structures.image_list import to_image_list

from ..backbone import build_backbone, build_flownet, build_embednet
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads


class GeneralizedRCNNDFF(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(GeneralizedRCNNDFF, self).__init__()

        self.backbone = build_backbone(cfg)
        self.flownet = build_flownet(cfg)
        self.rpn = build_rpn(cfg, self.backbone.out_channels)
        self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)

        self.device = cfg.MODEL.DEVICE

        self.key_images = None
        self.key_feats = None

    def get_grid(self, flow):
        m, n = flow.shape[-2:]
        shifts_x = torch.arange(0, n, 1, dtype=torch.float32, device=flow.device)
        shifts_y = torch.arange(0, m, 1, dtype=torch.float32, device=flow.device)
        shifts_y, shifts_x = torch.meshgrid(shifts_y, shifts_x)

        grid_dst = torch.stack((shifts_x, shifts_y)).unsqueeze(0)
        workspace = torch.tensor([(n - 1) / 2, (m - 1) / 2]).view(1, 2, 1, 1).to(flow.device)

        flow_grid = ((flow + grid_dst) / workspace - 1).permute(0, 2, 3, 1)

        return flow_grid

    def resample(self, feats, flow):
        flow_grid = self.get_grid(flow)
        warped_feats = F.grid_sample(feats, flow_grid, mode="bilinear", padding_mode="border")

        return warped_feats

    def forward(self, images, targets=None):
        """
        Arguments:
            #images (list[Tensor] or ImageList): images to be processed
            targets (list[BoxList]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        if self.training:
            images["cur"] = to_image_list(images["cur"])
            images["ref"] = [to_image_list(image) for image in images["ref"]]

            return self._forward_train(images["cur"], images["ref"], targets)
        else:
            images["cur"] = to_image_list(images["cur"])

            infos = images.copy()
            infos.pop("cur")
            return self._forward_test(images["cur"], infos)

    def _forward_train(self, img, imgs_ref, targets):
        img_cur, img_ref = img.tensors, imgs_ref[0].tensors
        feats_ref = self.backbone(img_ref)[0]

        # calculate flow and warp the feature
        concat_imgs_pair = torch.cat([img_cur / 255, img_ref / 255], dim=1)

        flow, scale_map = self.flownet(concat_imgs_pair)
        warped_feats_refs = self.resample(feats_ref, flow)

        # This part is different from the original implementation described in the paper.
        if True:
            feats = warped_feats_refs * scale_map
        else:
            feats = feats_ref
        feats = (feats, )

        proposals, proposal_losses = self.rpn(img, feats, targets)

        if self.roi_heads:
            x, result, detector_losses = self.roi_heads(feats, proposals, targets)
        else:
            detector_losses = {}

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses

    def _forward_test(self, imgs, infos, targets=None):
        if targets is not None:
            raise ValueError("In testing mode, targets should be None")

        if infos["is_key_frame"]:
            self.key_images = imgs
            self.key_feats = self.backbone(imgs.tensors)

        # The inference is also a slightly different from the original implementation.
        feats_ref = self.key_feats[0]
        flow, scale_map = self.flownet(torch.cat([imgs.tensors / 255, self.key_images.tensors / 255], dim=1))
        feats = self.resample(feats_ref, flow)
        feats = (feats * scale_map, )

        proposals, proposal_losses = self.rpn(imgs, feats, None)
        if self.roi_heads:
            x, result, detector_losses = self.roi_heads(feats, proposals, None)
        else:
            result = proposals

        return result
