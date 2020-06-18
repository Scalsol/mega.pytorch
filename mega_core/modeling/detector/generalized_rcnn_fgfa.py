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


class GeneralizedRCNNFGFA(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(GeneralizedRCNNFGFA, self).__init__()

        self.backbone = build_backbone(cfg)
        self.flownet = build_flownet(cfg)
        self.embednet = build_embednet(cfg)
        self.rpn = build_rpn(cfg, self.backbone.out_channels)
        self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)

        self.device = cfg.MODEL.DEVICE

        self.all_frame_interval = cfg.MODEL.VID.FGFA.ALL_FRAME_INTERVAL
        self.key_frame_location = cfg.MODEL.VID.FGFA.KEY_FRAME_LOCATION
        self.images = deque(maxlen=self.all_frame_interval)
        self.features = deque(maxlen=self.all_frame_interval)

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

    def compute_norm(self, embed):
        return torch.norm(embed, dim=1, keepdim=True) + 1e-10

    def compute_weight(self, embed_ref, embed_cur):
        embed_ref_norm = self.compute_norm(embed_ref)
        embed_cur_norm = self.compute_norm(embed_cur)

        embed_ref_normalized = embed_ref / embed_ref_norm
        embed_cur_normalized = embed_cur / embed_cur_norm

        weight = torch.sum(embed_ref_normalized * embed_cur_normalized, dim=1, keepdim=True)

        return weight

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
            images["ref"] = [to_image_list(image) for image in images["ref"]]

            infos = images.copy()
            infos.pop("cur")
            return self._forward_test(images["cur"], infos)

    def _forward_train(self, img, imgs_ref, targets):
        num_refs = len(imgs_ref)

        concat_imgs = torch.cat([img.tensors, *[img_ref.tensors for img_ref in imgs_ref]], dim=0)
        concat_feats = self.backbone(concat_imgs)[0]

        # calculate flow and warp the feature
        img_cur, imgs_ref = torch.split(concat_imgs, (1, num_refs), dim=0)
        img_cur_copies = img_cur.repeat(num_refs, 1, 1, 1)
        concat_imgs_pair = torch.cat([img_cur_copies / 255, imgs_ref / 255], dim=1)

        flow = self.flownet(concat_imgs_pair)

        feats_cur, feats_refs = torch.split(concat_feats, (1, num_refs), dim=0)
        warped_feats_refs = self.resample(feats_refs, flow)

        # calculate embedding and weights
        concat_feats = torch.cat([feats_cur, warped_feats_refs], dim=0)
        concat_embed_feats = self.embednet(concat_feats)
        embed_cur, embed_refs = torch.split(concat_embed_feats, (1, num_refs), dim=0)

        unnormalized_weights = self.compute_weight(embed_refs, embed_cur)
        weights = F.softmax(unnormalized_weights, dim=0)

        feats = (torch.sum(weights * warped_feats_refs, dim=0, keepdim=True), )
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
        """
        forward for the test phase.
        :param imgs:
        :param frame_category: 0 for start, 1 for normal
        :param targets:
        :return:
        """
        def update_feature(img=None, feats=None, embeds=None):
            if feats is None:
                feats = self.backbone(img)[0]
                embeds = self.embednet(feats)

            self.images.append(img)
            self.features.append(torch.cat([feats, embeds], dim=1))

        if targets is not None:
            raise ValueError("In testing mode, targets should be None")

        if infos["frame_category"] == 0:  # a new video
            self.seg_len = infos["seg_len"]
            self.end_id = 0

            self.images = deque(maxlen=self.all_frame_interval)
            self.features = deque(maxlen=self.all_frame_interval)

            feats_cur = self.backbone(imgs.tensors)[0]
            embeds_cur = self.embednet(feats_cur)
            while len(self.images) < self.key_frame_location + 1:
                update_feature(imgs.tensors, feats_cur, embeds_cur)

            while len(self.images) < self.all_frame_interval:
                self.end_id = min(self.end_id + 1, self.seg_len - 1)
                end_filename = infos["pattern"] % self.end_id
                end_image = Image.open(infos["img_dir"] % end_filename).convert("RGB")

                end_image = infos["transforms"](end_image)
                if isinstance(end_image, tuple):
                    end_image = end_image[0]
                end_image = end_image.view(1, *end_image.shape).to(self.device)

                update_feature(end_image)

        elif infos["frame_category"] == 1:
            self.end_id = min(self.end_id + 1, self.seg_len - 1)
            end_image = infos["ref"][0].tensors

            update_feature(end_image)

        all_images = torch.cat(list(self.images), dim=0)
        all_features = torch.cat(list(self.features), dim=0)

        cur_image = self.images[self.key_frame_location]
        cur_image_copies = cur_image.repeat(self.all_frame_interval, 1, 1, 1)
        concat_imgs_pair = torch.cat([cur_image_copies / 255, all_images / 255], dim=1)

        flow = self.flownet(concat_imgs_pair)
        warped_feats = self.resample(all_features, flow)

        warped_feats, embeds = torch.split(warped_feats, (1024, 2048), dim=1)

        # compute weights
        embed_cur = embeds[self.key_frame_location:self.key_frame_location + 1, :, :, :]

        unnormalized_weights = self.compute_weight(embeds.contiguous(), embed_cur)
        weights = F.softmax(unnormalized_weights, dim=0)

        feats = (torch.sum(weights * warped_feats, dim=0, keepdim=True), )

        proposals, proposal_losses = self.rpn(imgs, feats, None)
        if self.roi_heads:
            x, result, detector_losses = self.roi_heads(feats, proposals, None)
        else:
            result = proposals

        return result
