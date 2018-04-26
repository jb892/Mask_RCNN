import cv2
import os.path
import sys
import argparse
import tensorflow as tf
import time
import numpy as np

ROOT_DIR = os.getcwd()
sys.path.append(ROOT_DIR)

from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize

sys.path.append(os.path.join(ROOT_DIR, "samples/coco/"))
import coco

MODEL_DIR = os.path.join(ROOT_DIR, "logs")

COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)


config = InferenceConfig()
config.display()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)

class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

FLAGS = None

def PrintConfiguration():
    print("Tensorflow Version: ", tf.__version__)
    print("OpenCV Version: ", cv2.__version__)


def Filter(target_id, r):
    class_ids = r['class_ids']
    exclude_id = np.where(class_ids[:] != target_id)
    rois = np.delete(r['rois'], exclude_id, axis=0)
    class_ids = np.delete(r['class_ids'], exclude_id, axis=0)
    scores = np.delete(r['scores'], exclude_id, axis=0)
    return {
        'rois': rois,
        'class_ids': class_ids,
        'scores': scores,
    }

def VideoReader():
    videoCap = cv2.VideoCapture(FLAGS.path)
    success, image = videoCap.read()
    if success:
        start_time = time.time()
        results = model.detect([image], verbose=1)
        print("--- Detection time: %.2f s seconds ---" % round(time.time() - start_time, 2))
        r = Filter(1, results[0])
        print(r['rois'])
        print(r['class_ids'])
        print(r['scores'])

        visualize.display_instances_no_mask(image, r['rois'], r['class_ids'], class_names, r['scores'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--path',
        type=str,
        default='C:\\Users\\a\\Documents\\2DHumanCounterDemo\\countertest_baofeng.mp4',
        help='Location of the input MP4 video file need to be processed.'
    )
    parser.add_argument(
        '--bbox',
        type=bool,
        default=True,
        help='Whether show the bounding boxes.'
    )

    FLAGS, unparsed = parser.parse_known_args()

    PrintConfiguration()
    VideoReader()