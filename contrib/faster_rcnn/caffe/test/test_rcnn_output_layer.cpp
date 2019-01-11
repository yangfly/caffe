#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "gtest/gtest.h"

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layers/rcnn_output_layer.hpp"
#include "caffe/util/math_functions.hpp"

#include "caffe/test/test_caffe_main.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

using boost::scoped_ptr;

namespace caffe {

template <typename TypeParam>
class RCNNOutputLayerTest : public MultiDeviceTest<TypeParam> {
  typedef typename TypeParam::Dtype Dtype;

 protected:
  RCNNOutputLayerTest()
      : blob_bottom_bbox_(new Blob<Dtype>({2, 12})),
        blob_bottom_cls_(new Blob<Dtype>({2, 3})),
        blob_bottom_rois_(new Blob<Dtype>({2, 5})),
        blob_bottom_info_(new Blob<Dtype>({3})),
        blob_top_dets_(new Blob<Dtype>()) {
    // fill the bboxes
    vector<Dtype> delta = {-0.00006853, -0.00124648,  0.00270347, -0.00204985,
                           -0.07667218, -0.06754107, -0.1538136 , -0.14635995,
                           -0.02706362, -0.05281606, -0.244319  ,  0.01312275,
                           -0.00004825, -0.00113318,  0.00255779, -0.00119238,
                           -0.04818296, -0.04687576, -0.1697189 , -0.16557491,
                            0.04889064, -0.02887787, -0.364709  , -0.06754405};
    Dtype* bdata = blob_bottom_bbox_->mutable_cpu_data();
    caffe_copy(blob_bottom_bbox_->count(), delta.data(), bdata);
    blob_bottom_vec_.push_back(blob_bottom_bbox_);
    // fill the scores
    vector<Dtype> scores = {0.01, 0.44, 0.55,
                            0.12, 0.11, 0.77};
    Dtype* cdata = blob_bottom_cls_->mutable_cpu_data();
    caffe_copy(blob_bottom_cls_->count(), scores.data(), cdata);
    blob_bottom_vec_.push_back(blob_bottom_cls_);
    // fill the rois
    vector<Dtype> rois = {0, 166.21158 , 111.19214 , 790.75854 , 509.0631,
                          0,  38.520813,  69.31656 , 752.81616 , 549.2108};
    Dtype* rdata = blob_bottom_rois_->mutable_cpu_data();
    caffe_copy(blob_bottom_rois_->count(), rois.data(), rdata);
    blob_bottom_vec_.push_back(blob_bottom_rois_);
    // fill the info
    blob_bottom_info_->mutable_cpu_data()[0] = 600;
    blob_bottom_info_->mutable_cpu_data()[1] = 800;
    blob_bottom_info_->mutable_cpu_data()[2] = 1.6;
    blob_bottom_vec_.push_back(blob_bottom_info_);
    // add top dets
    blob_top_vec_.push_back(blob_top_dets_);
  }
  virtual ~RCNNOutputLayerTest() {
    delete blob_bottom_bbox_;
    delete blob_bottom_cls_;
    delete blob_bottom_rois_;
    delete blob_bottom_info_;
    delete blob_top_dets_;
  }
  Blob<Dtype>* const blob_bottom_bbox_;
  Blob<Dtype>* const blob_bottom_cls_;
  Blob<Dtype>* const blob_bottom_rois_;
  Blob<Dtype>* const blob_bottom_info_;
  Blob<Dtype>* const blob_top_dets_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

TYPED_TEST_CASE(RCNNOutputLayerTest, TestDtypesAndDevices);

TYPED_TEST(RCNNOutputLayerTest, TestForward) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  RCNNOutputParameter* rcnn_output_param =
      layer_param.mutable_rcnn_output_param();
  rcnn_output_param->set_conf_thresh(0.1);
  rcnn_output_param->set_nms_thresh(0.9);
  RCNNOutputLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
}

TYPED_TEST(RCNNOutputLayerTest, TestEmptyOutput) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  RCNNOutputParameter* rcnn_output_param =
      layer_param.mutable_rcnn_output_param();
  rcnn_output_param->set_conf_thresh(0.8);
  rcnn_output_param->set_nms_thresh(0.9);
  RCNNOutputLayer<Dtype> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, this->blob_top_vec_);
  layer.Forward(this->blob_bottom_vec_, this->blob_top_vec_);
  EXPECT_EQ(this->blob_top_dets_->count(), 1);
}

TYPED_TEST(RCNNOutputLayerTest, TestEmptyInput) {
  typedef typename TypeParam::Dtype Dtype;
  LayerParameter layer_param;
  RCNNOutputParameter* rcnn_output_param =
      layer_param.mutable_rcnn_output_param();
  rcnn_output_param->set_conf_thresh(0.1);
  rcnn_output_param->set_nms_thresh(0.9);
  RCNNOutputLayer<Dtype> layer(layer_param);
  vector<Blob<Dtype>*> blob_bottom_vec;
  blob_bottom_vec.push_back(new Blob<Dtype>({0, 324}));
  blob_bottom_vec.push_back(new Blob<Dtype>({0, 81}));
  blob_bottom_vec.push_back(new Blob<Dtype>({0, 5}));
  Blob<Dtype>* info = new Blob<Dtype>({3});
  info->mutable_cpu_data()[0] = 600;
  info->mutable_cpu_data()[1] = 800;
  info->mutable_cpu_data()[2] = 1.6;
  blob_bottom_vec.push_back(info);
  vector<Blob<Dtype>*> blob_top_vec;
  blob_top_vec.push_back(new Blob<Dtype>());
  layer.SetUp(blob_bottom_vec, blob_top_vec);
  layer.Forward(blob_bottom_vec, blob_top_vec);
  EXPECT_EQ(this->blob_top_dets_->count(), 0);
  for (auto p : blob_bottom_vec)
    delete p;
  for (auto p : blob_top_vec)
    delete p;
}

}  // namespace caffe
