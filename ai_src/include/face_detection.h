#ifndef FACE_DETECTION_H
#define FACE_DETECTION_H

#include "BYTETracker.h"
#include "face_alignment.h"
#include "head_pose_estimation.h"
#include "npu_model.h"
#include "types.h"
#include <algorithm>
#include <cassert>
#include <cstring>
#include <iostream>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <vector>

class FaceDetector : public NPUModel {
public:
  FaceDetector(const char *model_path, float obj_threshold = 0.5,
               float nms_threshold = 0.3, bool using_tracking = true);
  ~FaceDetector();



public:
  std::vector<types::FaceDetectRes> Detect(const cv::Mat &img);
  void AlignFace(const cv::Mat &image, types::FaceDetectRes &face_detect_res);
  void EstimateHeadPose(types::FaceDetectRes &face_detect_res);
  void SetCameraSize(int camera_width, int camera_height);
  void DrawOutput(cv::Mat &img, std::vector<types::FaceDetectRes> &boxes,
                  bool landmark = false);

private:
  std::vector<std::vector<float>> GenerateAnchorCenters(int stride);
  void GenerateProposals(const std::vector<std::vector<float>> &anchors,
                         int feat_stride, const std::vector<float> &score_blob,
                         const std::vector<float> &bbox_blob,
                         const std::vector<float> &kps_blob,
                         std::vector<types::FaceDetectRes> &face_objects);
  void NMS(std::vector<types::FaceDetectRes> &input_boxes);
  void FilterBoxes(std::vector<types::FaceDetectRes> &predicted_boxes);
  bool Preprocess(const cv::Mat &img, cv::Mat &resized_img,
                  float &scale) override;
  void Postprocess(std::vector<std::vector<float>> &outputs, const cv::Mat &img,
                   std::vector<types::FaceDetectRes> &face_detections,
                   float &scale);

private:
  FaceAligner face_aligner;
  HeadPoseEstimator head_pose_estimator;
  BYTETracker *face_tracker;

  float obj_threshold_;
  float nms_threshold_;
  bool using_tracking_;
  std::vector<int> list_stride_;
  std::vector<std::vector<std::vector<float>>> all_anchor_centers_;
};

#endif // FACE_DETECTION_H