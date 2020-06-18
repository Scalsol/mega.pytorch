import logging

from .vid_eval import do_vid_evaluation


def vid_evaluation(dataset, predictions, output_folder, box_only, motion_specific, **_):
    logger = logging.getLogger("mega_core.inference")
    logger.info("performing vid evaluation, ignored iou_types.")
    return do_vid_evaluation(
        dataset=dataset,
        predictions=predictions,
        output_folder=output_folder,
        box_only=box_only,
        motion_specific=motion_specific,
        logger=logger,
    )
