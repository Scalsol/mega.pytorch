# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""
import time

from PIL import Image
from collections import deque

import torch
from torch import nn

from mega_core.structures.image_list import to_image_list
from mega_core.structures.boxlist_ops import cat_boxlist

from ..backbone import build_backbone
from ..rpn.rpn import build_rpn
from ..roi_heads.roi_heads import build_roi_heads


class GeneralizedRCNNRDN(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(GeneralizedRCNNRDN, self).__init__()
        self.device = cfg.MODEL.DEVICE

        self.backbone = build_backbone(cfg)
        self.rpn = build_rpn(cfg, self.backbone.out_channels)
        self.roi_heads = build_roi_heads(cfg, self.backbone.out_channels)

        # enable advanced stage in the paper
        # self.advanced = cfg.MODEL.VID.ROI_BOX_HEAD.ATTENTION.ADVANCED_STAGE > 0
        # self.advanced_num = int(cfg.MODEL.VID.RPN.REF_POST_NMS_TOP_N * cfg.MODEL.VID.RDN.RATIO)

        self.all_frame_interval = cfg.MODEL.VID.RDN.ALL_FRAME_INTERVAL
        self.key_frame_location = cfg.MODEL.VID.RDN.KEY_FRAME_LOCATION

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

    def _forward_train(self, img_cur, imgs_ref, targets):
        concat_imgs = torch.cat([img_cur.tensors, *[img_ref.tensors for img_ref in imgs_ref]], dim=0)
        concat_feats = self.backbone(concat_imgs)[0]

        num_imgs = 1 + len(imgs_ref)

        feats_list = torch.chunk(concat_feats, num_imgs, dim=0)

        # propagate the features
        proposals_list = []
        # key frame
        proposals, proposal_losses = self.rpn(img_cur, (feats_list[0],), targets, version="key")
        proposals_list.append(proposals)

        # ref frame, notice the current frame could by treated as ref frame
        proposals_cur = self.rpn(img_cur, (feats_list[0], ), version="ref")
        proposals_list.append(proposals_cur[0])
        for i in range(len(imgs_ref)):
            proposals_ref = self.rpn(imgs_ref[i], (feats_list[i + 1], ), version="ref")
            proposals_list.append(proposals_ref[0])

        if self.roi_heads:
            x, result, detector_losses = self.roi_heads(feats_list,
                                                        proposals_list,
                                                        targets)
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
        def update_feature(img=None, feats=None, proposals=None, proposals_feat=None):
            assert (img is not None) or (feats is not None and proposals is not None and proposals_feat is not None)

            if img is not None:
                feats = self.backbone(img)[0]
                # note here it is `imgs`! for we only need its shape, it would not cause error, but is not explicit.
                proposals = self.rpn(imgs, (feats,), version="ref")
                proposals_feat = self.roi_heads.box.feature_extractor(feats, proposals, pre_calculate=True)

            self.feats.append(feats)
            self.proposals.append(proposals[0])
            self.proposals_feat.append(proposals_feat)
            # if self.advanced:
            #     self.proposals_adv.append(proposals[0][:self.advanced_num])
            #     self.proposals_feat_dis.append(proposals_feat[:self.advanced_num])

        if targets is not None:
            raise ValueError("In testing mode, targets should be None")

        if infos["frame_category"] == 0:  # a new video
            self.seg_len = infos["seg_len"]
            self.end_id = 0

            self.feats = deque(maxlen=self.all_frame_interval)
            self.proposals = deque(maxlen=self.all_frame_interval)
            self.proposals_feat = deque(maxlen=self.all_frame_interval)

            # if self.advanced:
            #     self.proposals_adv = deque(maxlen=self.all_frame_interval)
            #     self.proposals_feat_adv = deque(maxlen=self.all_frame_interval)

            feats_cur = self.backbone(imgs.tensors)[0]
            proposals_cur = self.rpn(imgs, (feats_cur, ), version="ref")
            proposals_feat_cur = self.roi_heads.box.feature_extractor(feats_cur, proposals_cur, pre_calculate=True)
            while len(self.feats) < self.key_frame_location + 1:
                update_feature(None, feats_cur, proposals_cur, proposals_feat_cur)

            while len(self.feats) < self.all_frame_interval:
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

        feats = self.feats[self.key_frame_location]
        proposals, proposal_losses = self.rpn(imgs, (feats, ), None)

        proposals_ref = cat_boxlist(list(self.proposals))
        proposals_feat_ref = torch.cat(list(self.proposals_feat), dim=0)

        # if self.advanced:
        #     proposals_ref_adv = cat_boxlist(list(self.proposals_adv))
        #     proposals_feat_ref_adv = torch.cat(list(self.proposals_feat_adv), dim=0)
        #
        #     proposals_list = [proposals, proposals_ref, proposals_feat_ref, proposals_ref_adv, proposals_feat_ref_adv]
        # else:
        proposals_list = [proposals, proposals_ref, proposals_feat_ref]

        if self.roi_heads:
            x, result, detector_losses = self.roi_heads(feats, proposals_list, None)
        else:
            result = proposals

        return result
