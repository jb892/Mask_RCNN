import cv2
import os.path
import sys
import argparse
import tensorflow as tf
import time
import numpy as np
import matplotlib.pyplot as plt

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
    # Read Video from a source video file.
    videoCap = cv2.VideoCapture(FLAGS.path)
    ret = True
    while (ret):
        # Read frame from the video source
        ret, frame = videoCap.read()
        if ret:
            results = model.detect([frame], verbose=0)

            # Filter out all detected objects that is not a person(index == 1).
            r = Filter(1, results[0])

            # Draw bbox
            if FLAGS.bbox:
                for b in r['rois']:
                    y1, x1, y2, x2 = b
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)

            # Draw Caption
            for i, s in enumerate(r['scores']):
                cv2.putText(frame, ("Person {:.3f}".format(s)), (r['rois'][i][1], r['rois'][i][0]-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1, cv2.LINE_AA)

            height, width = frame.shape[:2]
            # Draw Person Count
            cv2.putText(frame, ("Person Count: {}".format(len(r['class_ids']))), (10, height - 25), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 2, cv2.LINE_AA)

            # Show Image
            cv2.imshow("2D Human Counter Demo", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

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

    VideoReader()