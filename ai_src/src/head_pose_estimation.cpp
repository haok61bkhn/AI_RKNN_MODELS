#include "head_pose_estimation.h"

HeadPoseEstimator::HeadPoseEstimator(int camera_width, int camera_height)
    : camera_width_(camera_width), camera_height_(camera_height) {
  SetCameraSize(camera_width, camera_height);
}

HeadPoseEstimator::~HeadPoseEstimator() {}

void HeadPoseEstimator::SetCameraSize(int camera_width, int camera_height) {
  camera_width_ = camera_width;
  camera_height_ = camera_height;
  cv::Point2d camera_center(camera_width_ / 2.0, camera_height_ / 2.0);
  double focal_length = camera_center.x / std::tan(60.0 / 2.0 * M_PI / 180.0);
  camera_matrix_ = (cv::Mat_<double>(3, 3) << focal_length, 0, camera_center.x,
                    0, focal_length, camera_center.y, 0, 0, 1);
}

types::Orientation
HeadPoseEstimator::EstimateHeadPose(const std::vector<types::Point> &landmark) {

  cv::Mat dist_coeffs = cv::Mat::zeros(4, 1, CV_64F);

  double v_cx = landmark[2].x - (landmark[0].x + landmark[1].x) / 2.0;
  double v_cy = landmark[2].y - (landmark[0].y + landmark[1].y) / 2.0;

  std::vector<cv::Point2d> image_points = {
      cv::Point2d(landmark[2].x, landmark[2].y), // Nose tip
      cv::Point2d(landmark[2].x + 1.2 * v_cx,
                  landmark[2].y + 1.2 * v_cy),   // Chin
      cv::Point2d(landmark[0].x, landmark[0].y), // Left eye left corner
      cv::Point2d(landmark[1].x, landmark[1].y), // Right eye right corner
      cv::Point2d(landmark[3].x, landmark[3].y), // Left mouth corner
      cv::Point2d(landmark[4].x, landmark[4].y)  // Right mouth corner
  };

  cv::Mat rotation_vector, translation_vector;
  bool success = cv::solvePnP(model_points, image_points, camera_matrix_,
                              dist_coeffs, rotation_vector, translation_vector);

  types::Orientation result;

  if (success) {
    cv::Mat rotation_matrix;
    cv::Rodrigues(rotation_vector, rotation_matrix);

    cv::Mat projection_matrix(3, 4, CV_64F);
    rotation_matrix.copyTo(projection_matrix.colRange(0, 3));
    translation_vector.copyTo(projection_matrix.col(3));

    cv::Mat camera_matrix_out, rotation_matrix_out, translation_vector_out;
    cv::Mat euler_angles;
    cv::decomposeProjectionMatrix(projection_matrix, camera_matrix_out,
                                  rotation_matrix_out, translation_vector_out,
                                  cv::noArray(), cv::noArray(), cv::noArray(),
                                  euler_angles);

    double pitch = euler_angles.at<double>(0, 0) * M_PI / 180.0;
    double yaw = euler_angles.at<double>(1, 0) * M_PI / 180.0;
    double roll = euler_angles.at<double>(2, 0) * M_PI / 180.0;

    result.pitch = std::asin(std::sin(pitch)) * 180.0 / M_PI;
    result.yaw = std::asin(std::sin(yaw)) * 180.0 / M_PI;
    result.roll = -std::asin(std::sin(roll)) * 180.0 / M_PI;
  } else {
    result.pitch = 0.0;
    result.yaw = 0.0;
    result.roll = 0.0;
  }

  return result;
}