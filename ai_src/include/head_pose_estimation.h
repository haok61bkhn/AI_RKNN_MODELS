#ifndef HEAD_POSE_H
#define HEAD_POSE_H

#include "types.h"
#include <cmath>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>

class HeadPoseEstimator {
public:
  HeadPoseEstimator(int camera_width = 640, int camera_height = 480);
  ~HeadPoseEstimator();

public:
  void SetCameraSize(int camera_width, int camera_height);
  types::Orientation
  EstimateHeadPose(const std::vector<types::Point> &landmark);

private:
  int camera_width_;
  int camera_height_;
  cv::Mat camera_matrix_;
  std::vector<cv::Point3d> model_points = {
      cv::Point3d(0.0, 0.0, 0.0),          // Nose tip
      cv::Point3d(0.0, -330.0, -65.0),     // Chin
      cv::Point3d(-225.0, 170.0, -135.0),  // Left eye left corner
      cv::Point3d(225.0, 170.0, -135.0),   // Right eye right corner
      cv::Point3d(-150.0, -150.0, -125.0), // Mouth left corner
      cv::Point3d(150.0, -150.0, -125.0)   // Mouth right corner
  };
};

#endif // HEAD_POSE_H