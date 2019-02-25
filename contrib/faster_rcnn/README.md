# Faster RCNN

## Caffe

### layers

|      Layer       | Forward | Backward |      Test      |                                                      Source                                                      |
| :--------------: | :-----: | :------: | :------------: | ---------------------------------------------------------------------------------------------------------------- |
|  **ROIPooling**  |  C/GPU  |   GPU    |    :smiley:    | [caffe-fast-rcnn](https://github.com/rbgirshick/caffe-fast-rcnn/commit/0dcd397b29507b8314e252e850518c5695efbb83) |
| **SmoothL1Loss** |  C/GPU  |  C/GPU   |    :smiley:    | [caffe-fast-rcnn](https://github.com/rbgirshick/caffe-fast-rcnn/commit/0dcd397b29507b8314e252e850518c5695efbb83) |
|   **Proposal**   |  C/GPU  | no need  | :neutral_face: | [SeqFace](https://github.com/huangyangyu/SeqFace)                                                                |
|  **RCNNOutput**  |  C/GPU  | no need  |    :smiley:    | <font color=pink>own</font>                                                                                      |
> SmoothL1Loss has been enhanced (cpu forward/backward with test) for ssd.
> For layer usage please check tutorail/models/coco_vgg16_faster_rcnn_deploy.prototxt

### Util

- **nms**: cpu/gpu nms used in **ProposalLayer** and **RCNNOutputLayer**

## Tutorail

1. download faster-rcnn coco model to `models/`

```shell
$ ./models/fetch_coco_model.sh
```

2. run demo (on gpu 0)

```shell
$ python demo.py
```
optional `--gpu -1` to run on cpu.

## More

py-faster-rcnn: https://github.com/rbgirshick/py-faster-rcnn



