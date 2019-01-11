#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/" && pwd )"
cd $DIR

FILE=models/coco_vgg16_faster_rcnn_final.caffemodel
URL=https://dl.dropboxusercontent.com/s/cotx0y81zvbbhnt/coco_vgg16_faster_rcnn_final.caffemodel?dl=0
CHECKSUM=c576a1ee676abee9cafc4cf87d627ef3

## Downloading models
if [ ! -f $FILE ]; then
  echo "Downloading Faster R-CNN COCO model (528M)..."
  wget $URL -O $FILE
else
   echo "File already exists."
fi

## Checksum
if [ -f $FILE ]; then
  os=`uname -s`
  if [ "$os" = "Linux" ]; then
    checksum=`md5sum $FILE | awk '{ print $1 }'`
  elif [ "$os" = "Darwin" ]; then
    checksum=`cat $FILE | md5`
  fi
  if [ "$checksum" = "$CHECKSUM" ]; then
    echo "Checksum is correct."
  else
    echo "Checksum is incorrect."
    echo "Re-download model by remove $FILE and run $0 again."
    exit 0
  fi
else
  exit 0
fi

echo "Model: $FILE"