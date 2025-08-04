#include "face_replay_checker.h"

FaceReplayChecker::FaceReplayChecker(const char *model_path,
                                     const int &image_size,
                                     const float &threshold)
    : NPUModel(model_path) {
  input_width_ = image_size;
  input_height_ = image_size;
  threshold_ = threshold;
  if (!IsInitialized()) {
    return;
  }
}

FaceReplayChecker::~FaceReplayChecker() {}

void FaceReplayChecker::Preprocess(const cv::Mat &img, cv::Mat &processed_img) {
  cv::Mat resized_img;
  int old_width = img.cols;
  int old_height = img.rows;
  float ratio =
      static_cast<float>(input_width_) / std::max(old_width, old_height);
  int new_width = static_cast<int>(old_width * ratio);
  int new_height = static_cast<int>(old_height * ratio);
  cv::resize(img, resized_img, cv::Size(new_width, new_height));
  int delta_w = input_width_ - new_width;
  int delta_h = input_height_ - new_height;
  int top = delta_h / 2;
  int bottom = delta_h - top;
  int left = delta_w / 2;
  int right = delta_w - left;
  cv::copyMakeBorder(resized_img, processed_img, top, bottom, left, right,
                     cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
  cv::cvtColor(processed_img, processed_img, cv::COLOR_BGR2RGB);
};

cv::Mat FaceReplayChecker::IncreasedCrop(const cv::Mat &img,
                                         const cv::Rect &face_det,
                                         const float &bbox_inc) {
  int real_w = img.cols;
  int real_h = img.rows;
  int x = face_det.x;
  int y = face_det.y;
  int w = face_det.width;
  int h = face_det.height;
  int l = std::max(w, h);
  float xc = x + w / 2.0f;
  float yc = y + h / 2.0f;
  x = static_cast<int>(xc - l * bbox_inc / 2.0f);
  y = static_cast<int>(yc - l * bbox_inc / 2.0f);

  int x1 = (x < 0) ? 0 : x;
  int y1 = (y < 0) ? 0 : y;
  int x2 = (x + static_cast<int>(l * bbox_inc) > real_w)
               ? real_w
               : x + static_cast<int>(l * bbox_inc);
  int y2 = (y + static_cast<int>(l * bbox_inc) > real_h)
               ? real_h
               : y + static_cast<int>(l * bbox_inc);

  cv::Mat cropped_img = img(cv::Rect(x1, y1, x2 - x1, y2 - y1)).clone();

  int top = y1 - y;
  int bottom = static_cast<int>(l * bbox_inc - (y2 - y));
  int left = x1 - x;
  int right = static_cast<int>(l * bbox_inc - (x2 - x));

  cv::copyMakeBorder(cropped_img, cropped_img, top, bottom, left, right,
                     cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

  return cropped_img;
}

std::vector<float> FaceReplayChecker::Softmax(const std::vector<float> &input) {
  std::vector<float> output(input.size());
  float sum = 0.0;
  for (int i = 0; i < input.size(); ++i) {
    output[i] = exp(input[i]);
    sum += output[i];
  }
  for (int i = 0; i < output.size(); ++i) {
    output[i] /= sum;
  }
  return output;
}
bool FaceReplayChecker::Predict(const cv::Mat &img, const cv::Rect &face_det) {
  cv::Mat processed_img;
  cv::Mat cropped_img = IncreasedCrop(img, face_det, 1.5);
  Preprocess(cropped_img, processed_img);
  std::vector<std::vector<float>> output = Infer(processed_img);
  std::vector<float> pred = Softmax(output[0]);
  float score = pred[0];
  int max_index = 0;
  for (int j = 1; j < pred.size(); ++j) {
    if (pred[j] > score) {
      score = pred[j];
      max_index = j;
    }
  }
  if (max_index == 0 && score > threshold_) {
    return true;
  } else {
    return false;
  }
}