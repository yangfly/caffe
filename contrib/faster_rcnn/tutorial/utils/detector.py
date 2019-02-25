import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt

from .caffe_path import caffe
from .timer import Timer

__all__ = ['Detector']

# VOC Class list
CLASSES = dict(
    voc = ('aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
           'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
           'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'),
    coco = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
            'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
            'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
            'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
            'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
            'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
            'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
            'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
            'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
            'scissors', 'teddy bear', 'hair drier', 'toothbrush')
)

class Detector(object):
    """Faster R-CNN Detector"""
    def __init__(self, prototxt, caffemodel, gpu_id, dataset='coco',
                 scale=600, max_size=1000, transpose=(2, 0, 1),
                 mean=[102.9801, 115.9465, 122.7717]):
        if gpu_id < 0:
            caffe.set_mode_cpu()
        else:
            caffe.set_mode_gpu()
            caffe.set_device(gpu_id)
        self.net = caffe.Net(prototxt, caffe.TEST, weights=caffemodel)
        print('[{name}] Loaded network {model}'.format(
                    name=self.__class__.__name__, model=caffemodel))

        self.scale = scale
        self.max_size = max_size
        self.transpose = transpose
        self.mean = np.array(mean, dtype=np.float32)[None,None,:]
        self.classes = CLASSES[dataset]

        # colormap for visualization
        self.colormap = []
        for i in range(len(self.classes)):
            self.colormap.append(plt.get_cmap('hsv')(i / len(self.classes)))

    def preprocess(self, im):
        im = im.astype(np.float32) - self.mean
        short_size, long_size = sorted(im.shape[:2])
        factor = min(self.scale/short_size, self.max_size/long_size)
        im = cv2.resize(im, None, None, fx=factor, fy=factor)
        im = im.transpose(self.transpose)
        info = np.array((im.shape[1], im.shape[2], factor), dtype=np.float32)
        return im, info, factor

    def detect(self, im):
        im, info, factor = self.preprocess(im)
        self.net.blobs['data'].reshape(1, *(im.shape))
        self.net.blobs['data'].data[0,...] = im
        self.net.blobs['im_info'].data[...] = info
        dets = self.net.forward()['rcnn_out']
        if dets.ndim != 2:
            return np.empty((0,6), dtype=np.float32)
        else:
            return dets

    def demo(self, image):
        im = cv2.imread(image)
        timer = Timer()
        timer.tic()
        dets = self.detect(im)
        timer.toc()
        print ('Detection took {:.3f}s for {:d} objects'.format(timer.total_time, len(dets)))
        return self.plot(im, dets)
    
    def plot(self, im, dets, thresh=0, ax=None, linewidth=2.5):
        # create image axes
        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
        im = im[:, :, (2, 1, 0)] # to rgb
        ax.imshow(im.astype(np.uint8))
        if len(dets) == 0:
            return ax

        print(dets.shape)
        for det in dets:
            score = det[1]
            if score < thresh:
                continue
            class_id = int(det[0])
            x, y = det[2:4]
            w, h = det[4:6] - det[2:4]
            rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor=self.colormap[class_id], linewidth=linewidth)
            ax.add_patch(rect)
            ax.text(x, y-2, '{:s} {:.3f}'.format(self.classes[class_id], score),
                    bbox=dict(facecolor=self.colormap[class_id], alpha=0.5), fontsize=12, color='white')
        return ax