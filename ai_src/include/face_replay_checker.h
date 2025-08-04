#ifndef FACE_REPLAY_CHECKER_H
#define FACE_REPLAY_CHECKER_H
#include "npu_model.h"
#include "types.h"
#include <opencv2/opencv.hpp>

class FaceReplayChecker : public NPUModel {
public:
  FaceReplayChecker(const char *model_path, const int &image_size,
                    const float &threshold);
  ~FaceReplayChecker();

  bool Predict(const cv::Mat &img, const cv::Rect &face_det);

private:
  void Preprocess(const cv::Mat &img, cv::Mat &processed_img);
  cv::Mat IncreasedCrop(const cv::Mat &img, const cv::Rect &face_det,
                        const float &bbox_inc = 1.5);
  std::vector<float> Softmax(const std::vector<float> &input);

private:
  int batch_size_ = 1;
  float threshold_;
  int input_width_;
  int input_height_;
};
#endif // FACE_REPLAY_CHECKER_H