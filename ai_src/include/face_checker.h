#ifndef FACE_CHECKER_H
#define FACE_CHECKER_H

#include "face_antispoofing_checker.h"
#include "face_replay_checker.h"
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

class FaceChecker {
public:
  FaceChecker(
      const std::vector<std::string> &face_antipoofing_checker_model_paths,
      const float &face_antipoofing_checker_threshold,
      const std::string &face_replay_checker_model_path,
      const int &face_replay_checker_image_size,
      const float &face_replay_checker_threshold);
  ~FaceChecker();

  bool CheckFace(const cv::Mat &image, const cv::Rect &face_det);

private:
  FaceAntispoofingChecker *face_antipoofing_checker_;
  FaceReplayChecker *face_replay_checker_;
};

#endif // FACE_CHECKER_H