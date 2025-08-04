#include "face_checker.h"

FaceChecker::FaceChecker(
    const std::vector<std::string> &face_antipoofing_checker_model_paths,
    const float &face_antipoofing_checker_threshold,
    const std::string &face_replay_checker_model_path,
    const int &face_replay_checker_image_size,
    const float &face_replay_checker_threshold) {
  face_antipoofing_checker_ = new FaceAntispoofingChecker(
      face_antipoofing_checker_model_paths, face_antipoofing_checker_threshold);

  face_replay_checker_ = new FaceReplayChecker(
      face_replay_checker_model_path.c_str(), face_replay_checker_image_size,
      face_replay_checker_threshold);
}

FaceChecker::~FaceChecker() {
  delete face_antipoofing_checker_;
  delete face_replay_checker_;
}

bool FaceChecker::CheckFace(const cv::Mat &image, const cv::Rect &face_det) {
  bool is_real_antipoofing =
      face_antipoofing_checker_->Predict(image, face_det);
  if (!is_real_antipoofing) {
    std::cout << "face anti spoofing is not real" << std::endl;
    return false;
  }
  bool is_real_replay = face_replay_checker_->Predict(image, face_det);
  if (!is_real_replay) {
    std::cout << "face replay is not real" << std::endl;
    return false;
  }
  return is_real_replay;
}