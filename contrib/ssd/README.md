# SSD: Single Shot MultiBox Detector

## Caffe

### layers

|         Layer         | Forward | Backward |      Test      |
| :-------------------: | :-----: | :------: | :------------: |
|   **AnnotatedData**   |   CPU   | no need  |    :smiley:    |
|     **VideoData**     |   CPU   | no need  |       -        |
|     **PriorBox**      |   CPU   | no need  | :neutral_face: |
|      **Permute**      |  C/GPU  |  C/GPU   |    :smiley:    |
|     **Normalize**     |  C/GPU  |  C/GPU   |    :smiley:    |
|   **MultiboxLoss**    |   CPU   |   CPU    |    :smiley:    |
|  **DetectionOutput**  |  C/GPU  | no need  |    :smiley:    |
| **DetectionEvaluate** |   CPU   |  C/GPU   |    :smiley:    |
> All layer adapted from [Official SSD](#SSD)

### Util

- **bbox_utils**: bbox, anchors, matcher, encoder, decoder, etc.
- **im_transforms**: transform functions for data augmentation.
- **sampler**: sampler for data augmentation.
- **ssd_io**: io tool function
- **ssd_transformer**: a more power data_transformer.

### tools

- **convert_annoset**:  convert images and annotations to **lmdb**.
- **create_label_map**: create labelmap.txt from map (name, id, display_name) for visualization.
- **get_iamge_size**: retrieves the sizes of a set of images.

## Tutorail

1. Prepare voc lmdb dataset.

```shell
# download VOC2007 and VOC2012 dataset
$ ./data/VOC0712/download_voc.sh
# create list files: trainval.txt, test.txt, and test_name_size.txt
$ ./data/VOC0712/create_list.sh
# create lmdb files in data/VOCdevkit/VOC0712/lmdb
```

2. Train your model and evaluate the model on the fly.

```shell
# It will create model definition files and save snapshot models in:
#   - models/VGGNet/VOC0712/SSD_300x300/
# and job file, log file, and the python script in:
#   - jobs/VGGNet/VOC0712/SSD_300x300/
# and save temporary evaluation results in:
#   - data/VOCdevkit/results/VOC2007/SSD_300x300/
# It should reach 77.* mAP at 120k iterations.
$ python examples/ssd/ssd_pascal.py
```
If you don't have time to train your model, you can download a pre-trained model at [here](https://drive.google.com/open?id=0BzKzrI_SkD1_WVVTSmQxU0dVRzA).

3. Evaluate the most recent snapshot.

```shell
# If you would like to test a model you trained, you can do:
$ python examples/ssd/score_ssd_pascal.py
```

4. Test your model using a video file.

```shell
# attach examples/videos/ILSVRC2015_train_00755001.mp4 to visualize detection output.
$ python examples/ssd/ssd_pascal_video.py
```

5.  Test your model using a webcam. Note: press `esc` to stop.

```shell
# If you would like to attach a webcam to a model you trained, you can do:
$ python examples/ssd/ssd_pascal_webcam.py
```

## More

<a neme="SSD"></a>Official SSD: https://github.com/weiliu89/caffe/tree/ssd


