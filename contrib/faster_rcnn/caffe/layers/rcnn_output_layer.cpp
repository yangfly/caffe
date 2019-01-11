#include "caffe/layers/rcnn_output_layer.hpp"
#include "caffe/layers/proposal_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/nms.hpp"

using std::max;
using std::min;

namespace caffe {

template <typename Dtype>
void transform_inv_clip(Dtype* boxes, const Dtype* rois,
                        const int num_box, const int num_class,
                        const Dtype im_w, const Dtype im_h)
{
  Dtype max_w = im_w - 1;
  Dtype max_h = im_h - 1;
  Dtype w, h, cx, cy, pcx, pcy, pw, ph;
  for (int i = 0; i < num_box; i++) {
    // corner to center
    w = rois[3] - rois[1] + 1;
    h = rois[4] - rois[2] + 1;
    cx = rois[1] + 0.5 * w;
    cy = rois[2] + 0.5 * h;
    for (int j = 0; j < num_class; j++) {
      // transform back
      pcx = boxes[1] * w + cx;
      pcy = boxes[2] * h + cy;
      pw = exp(boxes[3]) * w;
      ph = exp(boxes[4]) * h;
      // center to corner
      boxes[1] = pcx - 0.5 * pw;
      boxes[2] = pcy - 0.5 * ph;
      boxes[3] = pcx + 0.5 * pw;
      boxes[4] = pcy + 0.5 * ph; 
      // clip bbox to be within the image region
      boxes[1] = max((Dtype)0,  min(boxes[1],  max_w));
      boxes[2] = max((Dtype)0,  min(boxes[2],  max_h));
      boxes[3] = max((Dtype)0,  min(boxes[3],  max_w));
      boxes[4] = max((Dtype)0,  min(boxes[4],  max_h));
      boxes += 5;
    }
    rois += 5;
  }
}

template <typename Dtype>
void bbox_filter(const Blob<Dtype>* bboxes, Blob<Dtype>* filtered,
                 const int class_id, const Dtype thresh)
{
  vector<int> indices;
  int num_box = bboxes->shape(0);
  int num_class = bboxes->shape(1);
  const Dtype* bbox = bboxes->cpu_data();
  for (int i = 0; i < num_box; i++) {
    if (bbox[class_id * 5] >= thresh)
      indices.push_back(i);
    bbox += num_class * 5;
  }
  int num = indices.size();
  filtered->Reshape({num, 5});
  if (num > 0) {
    Dtype* filted = filtered->mutable_cpu_data();
    bbox = bboxes->cpu_data();
    for (int i : indices) {
      int offset = (i * num_class + class_id) * 5;
      filted[0] = bbox[offset+1]; // x1
      filted[1] = bbox[offset+2]; // y1
      filted[2] = bbox[offset+3]; // x2
      filted[3] = bbox[offset+4]; // y2
      filted[4] = bbox[offset]; // score
      filted += 5;
    }
  }
}

template <typename Dtype>
void RCNNOutputLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  RCNNOutputParameter param = this->layer_param_.rcnn_output_param();
  nms_thresh_ = param.nms_thresh();
  conf_thresh_ = param.conf_thresh();
  // [label, confidence, xmin, ymin, xmax, ymax]
  top[0]->Reshape({0, 6});
}

template <typename Dtype>
void RCNNOutputLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                                       const vector<Blob<Dtype>*>& top)
{
  // prepare a dummy output as pycaffe doesn't support empty data export.
  top[0]->Reshape({1});
  top[0]->mutable_cpu_data()[0] = 0;

  // case: empty input (no proposals) 
  if (bottom[0]->count() == 0)
    return;

  // bottoms: box_pred, cls_pred, rois, im_info
  const Dtype* box_pred = bottom[0]->cpu_data();
  const Dtype* cls_pred = bottom[1]->cpu_data();
  const Dtype* rois = bottom[2]->cpu_data();
  const Dtype* info = bottom[3]->cpu_data();

  const int num_box = bottom[0]->shape(0);
  const int num_class = bottom[1]->shape(1);
  const Dtype im_h = info[0];
  const Dtype im_w = info[1];
  const Dtype factor = info[2];

  // [score, x1, y1, x2, y2]
  Blob<Dtype> bboxes({num_box, num_class, 5});
  // copy box_pred to bboxes
  Dtype* bdata = bboxes.mutable_cpu_data() + 1;
  for (int i = 0; i < num_box; i++)
    for (int j = 0; j < num_class; j++) {
      caffe_copy(4, box_pred, bdata);
      box_pred += 4;
      bdata += 5;
    }
  // transform back bboxes
  bdata = bboxes.mutable_cpu_data();
  transform_inv_clip(bdata, rois, num_box, num_class, im_w, im_h);
  // scale back bboxes
  caffe_scal(bboxes.count(), (Dtype)1. / factor, bdata);
  // copy scores to bboxes
  for (int i = 0; i < num_box * num_class; i++)
    bdata[5*i] = cls_pred[i];
  // nms on every foreground class
  int num_result = 0;
  vector<Blob<Dtype>*> results;
  for (int c = 1; c < num_class; c++) {
    Blob<Dtype> filtered;
    bbox_filter(&bboxes, &filtered, c, conf_thresh_);
    int num_in = filtered.shape(0);
    if (num_in == 0)
      continue;
    sort_box(filtered.mutable_cpu_data(), 0, num_in - 1, num_in);
    Blob<int> indices({num_in});
    int num_out;
    const Dtype* filted = filtered.cpu_data();
    nms_cpu(num_in, filted, indices.mutable_cpu_data(),
              &num_out, 0, nms_thresh_, -1);
    CHECK_GT(num_out, 0);
    Blob<Dtype>* result = new Blob<Dtype>({num_out, 6});
    const int* idxes = indices.cpu_data();
    Dtype* res = result->mutable_cpu_data();
    for (int i = 0; i < num_out; i++) {
      int offset = idxes[i] * 5;
      res[0] = c - 1;
      res[1] = filted[offset+4];  // score
      res[2] = filted[offset];    // x1
      res[3] = filted[offset+1];  // y1
      res[4] = filted[offset+2];  // x2
      res[5] = filted[offset+3];  // y2
      res += 6;
    }
    results.push_back(result);
    num_result += num_out;
  }

  if (num_result == 0)
    return;

  // merge all results
  CHECK_GT(num_result, 0);
  top[0]->Reshape({num_result, 6});
  Dtype* det = top[0]->mutable_cpu_data();
  for (const Blob<Dtype>* result : results) {
    int count = result->count();
    caffe_copy(count, result->cpu_data(), det);
    det += count;
    delete result;
  }
}

template <typename Dtype>
void RCNNOutputLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(RCNNOutputLayer);
#endif

INSTANTIATE_CLASS(RCNNOutputLayer);
REGISTER_LAYER_CLASS(RCNNOutput);

}  // namespace caffe