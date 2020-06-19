# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from mega_core.utils.env import setup_environment  # noqa F401 isort:skip
from mega_core.utils.dist_env import init_dist

import argparse
import os

from mega_core.config import cfg
from mega_core.data import make_data_loader
from mega_core.engine.inference import inference_no_model
from mega_core.utils.logger import setup_logger
from mega_core.utils.miscellaneous import mkdir

# Check if we can enable mixed-precision via apex.amp
try:
    from apex import amp
except ImportError:
    raise ImportError('Use APEX for mixed precision via apex.amp')


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Inference")
    parser.add_argument(
        "--config-file",
        default="/private/home/fmassa/github/detectron.pytorch_v2/configs/e2e_faster_rcnn_R_50_C4_1x_caffe2.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--prediction-folder",
        help="The path to the prediction file to be evaluated.",
        default=None,
    )
    parser.add_argument(
        "--motion-specific",
        "-ms",
        action="store_true",
        help="if True, evaluate motion-specific iou"
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    BASE_CONFIG = "configs/BASE_RCNN_1gpu.yaml"
    cfg.merge_from_file(BASE_CONFIG)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    save_dir = ""
    logger = setup_logger("mega_core", save_dir, 0)

    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST
    if args.prediction_folder:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(args.prediction_folder, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder
    data_loaders_val = make_data_loader(cfg, is_train=False)
    for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
        inference_no_model(
            data_loader_val,
            iou_types=iou_types,
            motion_specific=args.motion_specific,
            box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=output_folder,

        )


if __name__ == "__main__":
    main()
