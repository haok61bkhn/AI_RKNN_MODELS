#include "face_antispoofing_checker.h"

FaceAntispoofingCheckerModel::FaceAntispoofingCheckerModel(
    const char *model_path, const float &scale, const int &batch_size,
    const bool &org_resize, const float &shift_x, const float &shift_y,
    const int &height, const int &width)
    : NPUModel(model_path) {
  config_.org_resize = org_resize;
  config_.shift_x = shift_x;
  config_.shift_y = shift_y;
  config_.scale = scale;
  config_.height = height;
  config_.width = width;
  if (!IsInitialized()) {
    return;
  }
}

FaceAntispoofingCheckerModel::~FaceAntispoofingCheckerModel() {}
std::vector<float>
FaceAntispoofingCheckerModel::Softmax(const std::vector<float> &pred) {
  std::vector<float> pred_softmax(pred.size());
  float sum = 0.0;
  for (int i = 0; i < pred.size(); i++) {
    pred_softmax[i] = exp(pred[i]);
    sum += pred_softmax[i];
  }
  for (int i = 0; i < pred.size(); i++) {
    pred_softmax[i] /= sum;
  }
  return pred_softmax;
}

void FaceAntispoofingCheckerModel::Preprocess(const cv::Mat &img,
                                              const cv::Rect &box,
                                              cv::Mat &processed_img) {
  cv::Mat roi;
  if (config_.org_resize) {
    cv::resize(img, roi, cv::Size(config_.width, config_.height));
  } else {
    cv::Rect rect = CalculateBox(box, img.cols, img.rows, config_);
    cv::resize(img(rect).clone(), roi, cv::Size(config_.width, config_.height));
  }
  cv::cvtColor(roi, processed_img, cv::COLOR_BGR2RGB);
}

void FaceAntispoofingCheckerModel::Predict(const cv::Mat &img,
                                           const cv::Rect &face_det,
                                           std::vector<float> &pred) {
  cv::Mat processed_img;
  Preprocess(img, face_det, processed_img);
  pred = Softmax(Infer(processed_img)[0]);
}

cv::Rect FaceAntispoofingCheckerModel::CalculateBox(
    const cv::Rect &box, int src_w, int src_h,
    types::FaceCheckerModelConfig &config) {
  int x = box.x;
  int y = box.y;
  int box_width = box.width;
  int box_height = box.height;

  int y2 = y + box_height;
  y = y + box_height * 0.18;
  if (y > src_h - 1)
    y = src_h - 1;
  box_height = y2 - y;

  float scale = std::min(
      static_cast<float>(src_h - 1) / box_height,
      std::min(static_cast<float>(src_w - 1) / box_width, config.scale));

  float new_width = box_width * scale;
  float new_height = box_height * scale;

  float center_x = box_width / 2.0f + x;
  float center_y = box_height / 2.0f + y;

  float left_top_x = center_x - new_width / 2.0f;
  float left_top_y = center_y - new_height / 2.0f;
  float right_bottom_x = center_x + new_width / 2.0f;
  float right_bottom_y = center_y + new_height / 2.0f;

  if (left_top_x < 0) {
    right_bottom_x -= left_top_x;
    left_top_x = 0;
  }
  if (left_top_y < 0) {
    right_bottom_y -= left_top_y;
    left_top_y = 0;
  }
  if (right_bottom_x > src_w - 1) {
    left_top_x -= right_bottom_x - (src_w - 1);
    right_bottom_x = src_w - 1;
  }
  if (right_bottom_y > src_h - 1) {
    left_top_y -= right_bottom_y - (src_h - 1);
    right_bottom_y = src_h - 1;
  }

  return cv::Rect(left_top_x, left_top_y, right_bottom_x - left_top_x + 1,
                  right_bottom_y - left_top_y + 1);
}

FaceAntispoofingChecker::FaceAntispoofingChecker(
    const std::vector<std::string> &model_paths, const float &threshold) {
  threshold_ = threshold;
  for (int i = 0; i < model_paths.size(); ++i) {
    std::string model_path = model_paths[i];
    float scale = scales_[i];
    models_.push_back(
        new FaceAntispoofingCheckerModel(model_path.c_str(), scale));
  }
}

FaceAntispoofingChecker::~FaceAntispoofingChecker() {}

bool FaceAntispoofingChecker::Predict(const cv::Mat &img,
                                      const cv::Rect &face_det) {
  std::vector<float> pred;
  pred.clear();
  pred.resize(3);
  for (int i = 0; i < num_models_; ++i) {
    std::vector<float> tmp_pred;
    models_[i]->Predict(img, face_det, tmp_pred);
    pred[0] += tmp_pred[0];
    pred[1] += tmp_pred[1];
    pred[2] += tmp_pred[2];
  }

  int max_id = 0;
  if (pred[1] > pred[max_id])
    max_id = 1;
  if (pred[2] > pred[max_id])
    max_id = 2;
  float ats_confidence = pred[max_id] / num_models_;
  std::string fake_or_real = max_id == 1 ? "real" : "fake";
  std::cout << "ats_confidence: " << ats_confidence << " " << fake_or_real
            << std::endl;
  if (max_id == 1 && ats_confidence > threshold_)
    return true;
  else
    return false;
}

void FaceAntispoofingChecker::DrawFaceChecker(
    cv::Mat &img, std::vector<types::FaceDetectRes> &faces) {
  for (int i = 0; i < faces.size(); ++i) {
    std::string text = "";
    if (faces[i].is_real_face)
      text = "real_";
    else
      text = "fake_";

    cv::putText(img, text + std::to_string(faces[i].ats_confidence),
                cv::Point(faces[i].x1, faces[i].y1 - 10),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1);
  }
}
