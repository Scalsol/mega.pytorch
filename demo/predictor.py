import cv2
import numpy as np
import glob
import os
import tempfile
from collections import OrderedDict
from tqdm import tqdm
import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F
from mega_core.modeling.detector import build_detection_model
from mega_core.utils.checkpoint import DetectronCheckpointer
from mega_core.structures.image_list import to_image_list

from cv2 import (CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT, CAP_PROP_FPS,
                 CAP_PROP_FRAME_COUNT, CAP_PROP_FOURCC,
                 CAP_PROP_POS_FRAMES, VideoWriter_fourcc)


class Cache(object):

    def __init__(self, capacity):
        self._cache = OrderedDict()
        self._capacity = int(capacity)
        if capacity <= 0:
            raise ValueError('capacity must be a positive integer')

    @property
    def capacity(self):
        return self._capacity

    @property
    def size(self):
        return len(self._cache)

    def put(self, key, val):
        if key in self._cache:
            return
        if len(self._cache) >= self.capacity:
            self._cache.popitem(last=False)
        self._cache[key] = val

    def get(self, key, default=None):
        val = self._cache[key] if key in self._cache else default
        return val


class VideoProcessor(object):
    def __init__(self, filename, cache_capacity=10):
        if filename is None:
            self._fps = 25
            self._only_output = True
        else:
            self._vcap = cv2.VideoCapture(filename)
            assert cache_capacity > 0
            self._cache = Cache(cache_capacity)
            self._position = 0
            # get basic info
            self._width = int(self._vcap.get(CAP_PROP_FRAME_WIDTH))
            self._height = int(self._vcap.get(CAP_PROP_FRAME_HEIGHT))
            self._fps = self._vcap.get(CAP_PROP_FPS)
            self._frame_cnt = int(self._vcap.get(CAP_PROP_FRAME_COUNT))
            self._fourcc = self._vcap.get(CAP_PROP_FOURCC)
            self._only_output = False
        self._output_video_name = "visualization.avi"

    @property
    def vcap(self):
        """:obj:`cv2.VideoCapture`: The raw VideoCapture object."""
        return self._vcap

    @property
    def opened(self):
        """bool: Indicate whether the video is opened."""
        return self._vcap.isOpened()

    @property
    def width(self):
        """int: Width of video frames."""
        return self._width

    @property
    def height(self):
        """int: Height of video frames."""
        return self._height

    @property
    def resolution(self):
        """tuple: Video resolution (width, height)."""
        return (self._width, self._height)

    @property
    def fps(self):
        """float: FPS of the video."""
        return self._fps

    @property
    def frame_cnt(self):
        """int: Total frames of the video."""
        return self._frame_cnt

    @property
    def fourcc(self):
        """str: "Four character code" of the video."""
        return self._fourcc

    @property
    def position(self):
        """int: Current cursor position, indicating frame decoded."""
        return self._position

    def _get_real_position(self):
        return int(round(self._vcap.get(CAP_PROP_POS_FRAMES)))

    def _set_real_position(self, frame_id):
        self._vcap.set(CAP_PROP_POS_FRAMES, frame_id)
        pos = self._get_real_position()
        for _ in range(frame_id - pos):
            self._vcap.read()
        self._position = frame_id

    def read(self):
        """Read the next frame.

        If the next frame have been decoded before and in the cache, then
        return it directly, otherwise decode, cache and return it.

        Returns:
            ndarray or None: Return the frame if successful, otherwise None.
        """
        # pos = self._position
        if self._cache:
            img = self._cache.get(self._position)
            if img is not None:
                ret = True
            else:
                if self._position != self._get_real_position():
                    self._set_real_position(self._position)
                ret, img = self._vcap.read()
                if ret:
                    self._cache.put(self._position, img)
        else:
            ret, img = self._vcap.read()
        if ret:
            self._position += 1
        return img

    def get_frame(self, frame_id):
        """Get frame by index.

        Args:
            frame_id (int): Index of the expected frame, 0-based.

        Returns:
            ndarray or None: Return the frame if successful, otherwise None.
        """
        if frame_id < 0 or frame_id >= self._frame_cnt:
            raise IndexError(
                '"frame_id" must be between 0 and {}'.format(self._frame_cnt -
                                                             1))
        if frame_id == self._position:
            return self.read()
        if self._cache:
            img = self._cache.get(frame_id)
            if img is not None:
                self._position = frame_id + 1
                return img
        self._set_real_position(frame_id)
        ret, img = self._vcap.read()
        if ret:
            if self._cache:
                self._cache.put(self._position, img)
            self._position += 1
        return img

    def current_frame(self):
        """Get the current frame (frame that is just visited).

        Returns:
            ndarray or None: If the video is fresh, return None, otherwise
                return the frame.
        """
        if self._position == 0:
            return None
        return self._cache.get(self._position - 1)

    def cvt2frames(self,
                   frame_dir,
                   file_start=0,
                   filename_tmpl='{:06d}.jpg',
                   start=0,
                   max_num=0):
        """Convert a video to frame images

        Args:
            frame_dir (str): Output directory to store all the frame images.
            file_start (int): Filenames will start from the specified number.
            filename_tmpl (str): Filename template with the index as the
                placeholder.
            start (int): The starting frame index.
            max_num (int): Maximum number of frames to be written.
        """
        if not os.path.exists(frame_dir):
            os.makedirs(frame_dir)
        if max_num == 0:
            task_num = self.frame_cnt - start
        else:
            task_num = min(self.frame_cnt - start, max_num)
        if task_num <= 0:
            raise ValueError('start must be less than total frame number')
        if start > 0:
            self._set_real_position(start)

        for i in range(task_num):
            img = self.read()
            if img is None:
                break
            filename = os.path.join(frame_dir,
                                filename_tmpl.format(i + file_start))
            cv2.imwrite(filename, img)

    def frames2videos(self, frames, output_folder):
        if self._only_output:
            fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
            height, width = frames[0].shape[:2]
        else:
            fourcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
            height = self._height
            width = self._width

        videoWriter = cv2.VideoWriter(os.path.join(output_folder, self._output_video_name), fourcc, self._fps, (width, height))

        for frame_id in range(len(frames)):
            videoWriter.write(frames[frame_id])
        videoWriter.release()

    def __len__(self):
        return self.frame_cnt

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [
                self.get_frame(i)
                for i in range(*index.indices(self.frame_cnt))
            ]
        # support negative indexing
        if index < 0:
            index += self.frame_cnt
            if index < 0:
                raise IndexError('index out of range')
        return self.get_frame(index)

    def __iter__(self):
        self._set_real_position(0)
        return self

    def __next__(self):
        img = self.read()
        if img is not None:
            return img
        else:
            raise StopIteration

    next = __next__

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self._vcap.release()


class Resize(object):
    def __init__(self, min_size, max_size):
        self.min_size = min_size
        self.max_size = max_size

    # modified from torchvision to add support for max size
    def get_size(self, image_size):
        w, h = image_size
        size = self.min_size
        max_size = self.max_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def __call__(self, image):
        size = self.get_size(image.size)
        image = F.resize(image, size)
        return image


class VIDDemo(object):
    CATEGORIES = ['__background__',  # always index 0
                  'airplane', 'antelope', 'bear', 'bicycle',
                  'bird', 'bus', 'car', 'cattle',
                  'dog', 'domestic_cat', 'elephant', 'fox',
                  'giant_panda', 'hamster', 'horse', 'lion',
                  'lizard', 'monkey', 'motorcycle', 'rabbit',
                  'red_panda', 'sheep', 'snake', 'squirrel',
                  'tiger', 'train', 'turtle', 'watercraft',
                  'whale', 'zebra']

    def __init__(
            self,
            cfg,
            method="base",
            confidence_threshold=0.7,
            output_folder="demo/visulaization"
    ):
        self.cfg = cfg.clone()
        self.model = build_detection_model(cfg)
        self.model.eval()
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.model.to(self.device)

        save_dir = cfg.OUTPUT_DIR
        checkpointer = DetectronCheckpointer(cfg, self.model, save_dir=save_dir)
        _ = checkpointer.load(cfg.MODEL.WEIGHT)

        self.transforms = self.build_transform()

        # used to make colors for each class
        self.palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])

        self.cpu_device = torch.device("cpu")
        self.confidence_threshold = confidence_threshold

        self.method = method
        self.output_folder = output_folder
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        # for video output
        self.vprocessor = VideoProcessor(None)

    def build_transform(self):
        """
        Creates a basic transformation that was used to train the models
        """
        cfg = self.cfg

        # we are loading images with OpenCV, so we don't need to convert them
        # to BGR, they are already! So all we need to do is to normalize
        # by 255 if we want to convert to BGR255 format, or flip the channels
        # if we want it to be in RGB in [0-1] range.
        if cfg.INPUT.TO_BGR255:
            to_bgr_transform = T.Lambda(lambda x: x * 255)
        else:
            to_bgr_transform = T.Lambda(lambda x: x[[2, 1, 0]])

        normalize_transform = T.Normalize(
            mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
        )
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        transform = T.Compose(
            [
                T.ToPILImage(),
                Resize(min_size, max_size),
                T.ToTensor(),
                to_bgr_transform,
                normalize_transform,
            ]
        )
        return transform

    def build_pil_transform(self):
        """
        Creates a basic transformation that was used in generalized_rnn_{}._forward_test()
        """
        cfg = self.cfg

        # we are loading images with OpenCV, so we don't need to convert them
        # to BGR, they are already! So all we need to do is to normalize
        # by 255 if we want to convert to BGR255 format, or flip the channels
        # if we want it to be in RGB in [0-1] range.
        if cfg.INPUT.TO_BGR255:
            to_bgr_transform = T.Lambda(lambda x: x[[2, 1, 0]] * 255)
        else:
            to_bgr_transform = T.Lambda(lambda x: x)

        normalize_transform = T.Normalize(
            mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
        )
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        transform = T.Compose(
            [
                Resize(min_size, max_size),
                T.ToTensor(),
                to_bgr_transform,
                normalize_transform,
            ]
        )
        return transform

    def perform_transform(self, original_image):
        image = self.transforms(original_image)
        image_list = to_image_list(image, self.cfg.DATALOADER.SIZE_DIVISIBILITY)
        image_list = image_list.to(self.device)

        return image_list

    def run_on_image_folder(self, image_folder, suffix='.JPEG'):
        image_names = glob.glob(image_folder + '/*' + suffix)
        image_names = sorted(image_names)

        img_dir = "%s" + suffix
        frame_seg_len = len(image_names)
        pattern = image_folder + "/%06d"

        images_with_boxes = []

        for frame_id in tqdm(range(frame_seg_len)):
            original_image = cv2.imread(image_names[frame_id])
            img_cur = self.perform_transform(original_image)
            if self.method == "base":
                image_with_boxes = self.run_on_image(original_image, img_cur)
                images_with_boxes.append(image_with_boxes)
            elif self.method in ("dff", "fgfa", "rdn", "mega"):
                infos = {}
                infos["cur"] = img_cur
                infos["frame_category"] = 0 if frame_id == 0 else 1
                infos["seg_len"] = frame_seg_len
                infos["pattern"] = pattern
                infos["img_dir"] = img_dir
                infos["transforms"] = self.build_pil_transform()
                if self.method == "dff":
                    infos["is_key_frame"] = True if frame_id % 10 == 0 else False
                elif self.method in ("fgfa", "rdn"):
                    img_refs = []
                    if self.method == "fgfa":
                        max_offset = self.cfg.MODEL.VID.FGFA.MAX_OFFSET
                    else:
                        max_offset = self.cfg.MODEL.VID.RDN.MAX_OFFSET
                    ref_id = min(frame_seg_len - 1, frame_id + max_offset)
                    ref_filename = pattern % ref_id
                    img_ref = cv2.imread(img_dir % ref_filename)
                    img_ref = self.perform_transform(img_ref)
                    img_refs.append(img_ref)

                    infos["ref"] = img_refs
                elif self.method == "mega":
                    img_refs_l = []
                    # reading other images of the queue (not necessary to be the last one, but last one here)
                    ref_id = min(frame_seg_len - 1, frame_id + self.cfg.MODEL.VID.MEGA.MAX_OFFSET)
                    ref_filename = pattern % ref_id
                    img_ref = cv2.imread(img_dir % ref_filename)
                    img_ref = self.perform_transform(img_ref)
                    img_refs_l.append(img_ref)

                    img_refs_g = []
                    if self.cfg.MODEL.VID.MEGA.GLOBAL.ENABLE:
                        size = self.cfg.MODEL.VID.MEGA.GLOBAL.SIZE if frame_id == 0 else 1
                        shuffled_index = np.arange(frame_seg_len)
                        if self.cfg.MODEL.VID.MEGA.GLOBAL.SHUFFLE:
                            np.random.shuffle(shuffled_index)
                        for id in range(size):
                            filename = pattern % shuffled_index[
                                (frame_id + self.cfg.MODEL.VID.MEGA.GLOBAL.SIZE - id - 1) % frame_seg_len]
                            img = cv2.imread(img_dir % filename)
                            img = self.perform_transform(img)
                            img_refs_g.append(img)

                    infos["ref_l"] = img_refs_l
                    infos["ref_g"] = img_refs_g
                else:
                    pass
                image_with_boxes = self.run_on_image(original_image, infos)
                images_with_boxes.append(image_with_boxes)
            else:
                raise NotImplementedError("method {} is not implemented.".format(self.method))

        return images_with_boxes

    def run_on_video(self, video_path):
        if not os.path.isfile(video_path):
            raise FileNotFoundError('file "{}" does not exist'.format(video_path))
        self.vprocessor = VideoProcessor(video_path)
        tmpdir = tempfile.mkdtemp()
        self.vprocessor.cvt2frames(tmpdir)
        results = self.run_on_image_folder(tmpdir, suffix='.jpg')

        return results

    def run_on_image(self, image, infos=None):
        """
        Arguments:
            image
            infos
        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        predictions = self.compute_prediction(image, infos)
        top_predictions = self.select_top_predictions(predictions)

        result = image.copy()
        result = self.overlay_boxes(result, top_predictions)
        result = self.overlay_class_names(result, top_predictions)

        return result

    def compute_prediction(self, original_image, infos):
        """
        Arguments:
            original_image (np.ndarray): an image as returned by OpenCV
        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        # compute predictions
        with torch.no_grad():
            predictions = self.model(infos)
        predictions = [o.to(self.cpu_device) for o in predictions]

        # always single image is passed at a time
        prediction = predictions[0]

        # reshape prediction (a BoxList) into the original image size
        height, width = original_image.shape[:-1]
        prediction = prediction.resize((width, height))

        return prediction

    def select_top_predictions(self, predictions):
        """
        Select only predictions which have a `score` > self.confidence_threshold,
        and returns the predictions in descending order of score
        Arguments:
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `scores`.
        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        scores = predictions.get_field("scores")
        keep = torch.nonzero(scores > self.confidence_threshold).squeeze(1)
        predictions = predictions[keep]
        scores = predictions.get_field("scores")
        _, idx = scores.sort(0, descending=True)
        return predictions[idx]

    def compute_colors_for_labels(self, labels):
        """
        Simple function that adds fixed colors depending on the class
        """
        colors = labels[:, None] * self.palette
        colors = (colors % 255).numpy().astype("uint8")
        return colors

    def overlay_boxes(self, image, predictions):
        """
        Adds the predicted boxes on top of the image
        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `labels`.
        """
        labels = predictions.get_field("labels")
        boxes = predictions.bbox

        colors = self.compute_colors_for_labels(labels).tolist()

        for box, color in zip(boxes, colors):
            box = box.to(torch.int64)
            top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
            image = cv2.rectangle(
                image, tuple(top_left), tuple(bottom_right), tuple(color), 1
            )

        return image

    def overlay_class_names(self, image, predictions):
        """
        Adds detected class names and scores in the positions defined by the
        top-left corner of the predicted bounding box
        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `scores` and `labels`.
        """
        scores = predictions.get_field("scores").tolist()
        labels = predictions.get_field("labels").tolist()
        labels = [self.CATEGORIES[i] for i in labels]
        boxes = predictions.bbox

        template = "{}: {:.2f}"
        for box, score, label in zip(boxes, scores, labels):
            x, y = box[:2]
            s = template.format(label, score)
            cv2.putText(
                image, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
            )

        return image

    def generate_images(self, visualization_results):
        for frame_id in range(len(visualization_results)):
            cv2.imwrite(os.path.join(self.output_folder, "%06d.jpg" % frame_id), visualization_results[frame_id])

    def generate_video(self, visualization_results):
        self.vprocessor.frames2videos(visualization_results, self.output_folder)
