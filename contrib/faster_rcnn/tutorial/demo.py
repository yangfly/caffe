from __future__ import division

import os
os.environ['GLOG_minloglevel'] = '1'
import argparse
import numpy as np
import matplotlib.pyplot as plt
from utils import Detector

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', type=int, default=0,
                        help='GPU device id to use [0]')
    parser.add_argument('--data', dest='root', type=str, default="data",
                        help='directory of images ["data"]')
    parser.add_argument('--pt', dest='prototxt', type=str,
                        default='models/coco_vgg16_faster_rcnn_deploy.prototxt')
    parser.add_argument('--model', dest='caffemodel', type=str,
                        default='models/coco_vgg16_faster_rcnn_final.caffemodel')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    if not os.path.isfile(args.caffemodel):
        raise IOError(('{:s} not found.\nDid you run fetch_models.sh?').format(args.caffemodel))

    detector = Detector(args.prototxt, args.caffemodel, args.gpu_id)

    # Warmup on a dummy image
    im = 128 * np.ones((300, 500, 3), dtype=np.uint8)
    for i in range(2):
        detector.detect(im)

    im_names = ['000456.jpg', '000542.jpg', '001150.jpg',
                '001763.jpg', '004545.jpg']
    for im_name in im_names:
        im_file = os.path.join(args.root, im_name)
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('Demo for {}'.format(im_file))
        detector.demo(im_file)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()
    plt.show()
