#include "caffe/layers/rcnn_output_layer.hpp"
#include "caffe/layers/proposal_layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/nms.hpp"

namespace caffe {

template <typename Dtype>
__global__ static
void transform_inv_crop(const int num_box, const int num_class,
                        Dtype* bboxes, const Dtype* rois,
                        const Dtype max_w, const Dtype max_h)
{
  bboxes += (blockIdx.x * blockDim.x + threadIdx.x) * 5;
  rois += threadIdx.x * 5;
  // corner to center
  Dtype w = rois[3] - rois[1] + 1;
  Dtype h = rois[4] - rois[2] + 1;
  Dtype cx = rois[1] + 0.5 * w;
  Dtype cy = rois[2] + 0.5 * h;
  // transform back
  Dtype pcx = bboxes[0] * w + cx;
  Dtype pcy = bboxes[1] * h + cy;
  Dtype pw = exp(bboxes[2]) * w;
  Dtype ph = exp(bboxes[3]) * h;
  // center to corner
  bboxes[0] = pcx - 0.5 * pw;
  bboxes[1] = pcy - 0.5 * ph;
  bboxes[2] = pcx + 0.5 * pw;
  bboxes[3] = pcy + 0.5 * ph;
  // clip bbox to be within the image region
  bboxes[0] = max((Dtype)0,  min(bboxes[0],  max_w));
  bboxes[1] = max((Dtype)0,  min(bboxes[1],  max_h));
  bboxes[2] = max((Dtype)0,  min(bboxes[2],  max_w));
  bboxes[3] = max((Dtype)0,  min(bboxes[3],  max_h));
}

template <typename Dtype>
void bbox_filter(Dtype* bboxes, int& num,
                 const int num_box, const Dtype thresh)
{
  for (int i = 0; i < num_box; i++) {
    if (bboxes[i*5+4] >= thresh) {
      if (i > num) {
        bboxes[num*5]   = bboxes[i*5];  // x1
        bboxes[num*5+1] = bboxes[i*5+1];  // y1
        bboxes[num*5+2] = bboxes[i*5+2];  // x2
        bboxes[num*5+3] = bboxes[i*5+3];  // y2
        bboxes[num*5+4] = bboxes[i*5+4];  // score
      }
      num++;
    }
  }
}

// copy box_pred to bboxes
// layout: [n, c, 4] => [c, n, 5]
template <typename Dtype>
__global__ static
void box_transpose_copy(const int num_box, const int num_class,
                        Dtype* bboxes, const Dtype* box_pred)
{
  bboxes += (blockIdx.x * num_box + threadIdx.x) * 5;
  box_pred += (threadIdx.x * num_class + blockIdx.x) * 4;
  bboxes[0] = box_pred[0]; // x1
  bboxes[1] = box_pred[1]; // y1
  bboxes[2] = box_pred[2]; // x2
  bboxes[3] = box_pred[3]; // y2
}

template <typename Dtype>
__global__ static
void cls_transpose_copy(const int num_box, const int num_class,
                        Dtype* bboxes, const Dtype* cls_pred)
{
  bboxes[(blockIdx.x * num_box + threadIdx.x) * 5 + 4] 
      = cls_pred[threadIdx.x * num_class + blockIdx.x];
}

template <typename Dtype>
__global__ static
void all_result_copy(const int num_result, const int num_box,
                     Dtype* result, const Dtype* bboxes, const int* cood)
{
  cood += threadIdx.x * 2;
  bboxes += (cood[0] * num_box + cood[1]) * 5;
  result += threadIdx.x * 6;
  result[0] = cood[0] - 1;  // foreground class id
  result[1] = bboxes[4];  // score
  result[2] = bboxes[0];  // x1
  result[3] = bboxes[1];  // y1
  result[4] = bboxes[2];  // x2
  result[5] = bboxes[3];  // y2
}

template <typename Dtype>
void RCNNOutputLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top)
{
  // prepare a dummy output as pycaffe doesn't support empty data export.
  top[0]->Reshape({1});
  top[0]->mutable_cpu_data()[0] = 0;
  
  // case: empty input (no proposals) 
  if (bottom[0]->count() == 0)
    return;
  
  // bottoms: box_pred, cls_pred, rois, im_info
  const Dtype* box_pred = bottom[0]->gpu_data();
  const Dtype* cls_pred = bottom[1]->gpu_data();
  const Dtype* rois = bottom[2]->gpu_data();
  const Dtype* info = bottom[3]->cpu_data();

  const int num_box = bottom[0]->shape(0);
  const int num_class = bottom[1]->shape(1);
  const Dtype max_h = info[0] - 1;
  const Dtype max_w = info[1] - 1;
  const Dtype scale = 1. / info[2];

  // (c, n, 5) [score, x1, y1, x2, y2]
  Blob<Dtype> bboxes({num_class, num_box, 5});
  box_transpose_copy<Dtype><<<num_class, num_box>>>(
      num_box, num_class, bboxes.mutable_gpu_data(), box_pred);
  transform_inv_crop<Dtype><<<num_class, num_box>>>(
      num_box, num_class, bboxes.mutable_gpu_data(), rois, max_w, max_h);
  caffe_gpu_scal(bboxes.count(), scale, bboxes.mutable_gpu_data());
  cls_transpose_copy<Dtype><<<num_class, num_box>>>(
      num_box, num_class, bboxes.mutable_gpu_data(), cls_pred);

  // nms on every foreground class
  int num_result = 0;
  Blob<int> coords({num_box * num_class, 2});
  int* cood = coords.mutable_cpu_data();
  Blob<int> indices({num_box});
  Blob<int> mask;
  for (int c = 1; c < num_class; c++) {
    int num = 0;
    Dtype* bdata = bboxes.mutable_cpu_data() + c * num_box * 5;
    bbox_filter(bdata, num, num_box, conf_thresh_);
    sort_box(bdata, 0, num-1, num);
    if (num == 0)
      continue;
    nms_gpu(num, bboxes.gpu_data() + c * num_box * 5, &mask, 
            indices.mutable_cpu_data(), &num, 0, nms_thresh_, -1);
    for (int i = 0; i < num; i++) {
      cood[0] = c;
      cood[1] = indices.cpu_data()[i];
      cood += 2;
    }
    num_result += num;
  }
  if (num_result == 0)
    return;
  
  // copy all result
  CHECK_GT(num_result, 0);
  top[0]->Reshape({num_result, 6});
  all_result_copy<Dtype><<<1, num_result>>>(
      num_result, num_box, top[0]->mutable_gpu_data(),
      bboxes.gpu_data(), coords. gpu_data()
  );
}

template <typename Dtype>
void RCNNOutputLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}

INSTANTIATE_LAYER_GPU_FUNCS(RCNNOutputLayer);


}  // namespace caffe
