#include "obb_detection_yolov11.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <opencv2/opencv.hpp>

OBBDetectionYoloV11::OBBDetectionYoloV11(const char *model_path,
                                         const char *classes_file,
                                         float conf_threshold,
                                         float iou_threshold)
    : NPUModel(model_path), conf_threshold_(conf_threshold),
      iou_threshold_(iou_threshold) {
  if (!IsInitialized()) {
    return;
  }
  classes_ = ReadClasses(classes_file);
}

OBBDetectionYoloV11::~OBBDetectionYoloV11() {}

std::vector<std::string>
OBBDetectionYoloV11::ReadClasses(const std::string &filename) {
  std::ifstream file(filename);
  std::vector<std::string> result;
  if (file.is_open()) {
    std::string line;
    while (std::getline(file, line)) {
      result.push_back(line);
    }
    file.close();
  }
  return result;
}

float OBBDetectionYoloV11::Clamp(float val, float min, float max) {
  return val > min ? (val < max ? val : max) : min;
}

float OBBDetectionYoloV11::CalculateRotatedIoU(const std::vector<float> &box1,
                                               const std::vector<float> &box2) {
  float x1 = box1[0], y1 = box1[1], w1 = box1[2], h1 = box1[3], a1 = box1[4];
  float x2 = box2[0], y2 = box2[1], w2 = box2[2], h2 = box2[3], a2 = box2[4];

  float a1_var = w1 * w1 / 12.0f;
  float b1_var = h1 * h1 / 12.0f;
  float cos_a1 = std::cos(a1);
  float sin_a1 = std::sin(a1);
  float a1_cov = a1_var * cos_a1 * cos_a1 + b1_var * sin_a1 * sin_a1;
  float b1_cov = a1_var * sin_a1 * sin_a1 + b1_var * cos_a1 * cos_a1;
  float c1_cov = a1_var * cos_a1 * sin_a1 - b1_var * sin_a1 * cos_a1;

  float a2_var = w2 * w2 / 12.0f;
  float b2_var = h2 * h2 / 12.0f;
  float cos_a2 = std::cos(a2);
  float sin_a2 = std::sin(a2);
  float a2_cov = a2_var * cos_a2 * cos_a2 + b2_var * sin_a2 * sin_a2;
  float b2_cov = a2_var * sin_a2 * sin_a2 + b2_var * cos_a2 * cos_a2;
  float c2_cov = a2_var * cos_a2 * sin_a2 - b2_var * sin_a2 * cos_a2;

  float eps = 1e-7f;
  float dx = x1 - x2;
  float dy = y1 - y2;

  float a_sum = a1_cov + a2_cov;
  float b_sum = b1_cov + b2_cov;
  float c_sum = c1_cov + c2_cov;

  float det = a_sum * b_sum - c_sum * c_sum + eps;

  float t1 = (a_sum * dy * dy + b_sum * dx * dx) / det * 0.25f;
  float t2 = (c_sum * dx * dy) / det * 0.5f;

  float det1 = a1_cov * b1_cov - c1_cov * c1_cov;
  float det2 = a2_cov * b2_cov - c2_cov * c2_cov;
  det1 = std::max(det1, 0.0f);
  det2 = std::max(det2, 0.0f);

  float t3 =
      std::log((det + eps) / (4.0f * std::sqrt(det1 * det2) + eps) + eps) *
      0.5f;

  float bd = t1 + t2 + t3;
  bd = Clamp(bd, eps, 100.0f);
  float hd = std::sqrt(1.0f - std::exp(-bd) + eps);

  return 1.0f - hd;
}

bool OBBDetectionYoloV11::Preprocess(const cv::Mat &img, cv::Mat &resized_img,
                                     float &scale, int &pad_left,
                                     int &pad_top) {
  float height = img.rows;
  float width = img.cols;

  float r = std::min(input_height_ / height, input_width_ / width);
  int new_unpad_w = std::round(width * r);
  int new_unpad_h = std::round(height * r);

  scale = 1.0f / r;

  cv::Mat tmp;
  if (new_unpad_w != (int)width || new_unpad_h != (int)height) {
    cv::resize(img, tmp, cv::Size(new_unpad_w, new_unpad_h));
  } else {
    tmp = img.clone();
  }

  int pad_width = input_width_ - new_unpad_w;
  int pad_height = input_height_ - new_unpad_h;
  int top = pad_height / 2;
  int bottom = pad_height - top;
  int left = pad_width / 2;
  int right = pad_width - left;
  pad_left = left;
  pad_top = top;

  cv::copyMakeBorder(tmp, resized_img, top, bottom, left, right,
                     cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));
  cv::cvtColor(resized_img, resized_img, cv::COLOR_BGR2RGB);
  return true;
}

void OBBDetectionYoloV11::Postprocess(std::vector<std::vector<float>> &outputs,
                                      const cv::Mat &img,
                                      std::vector<types::OBB_DET> &detections,
                                      float &scale, int &pad_left,
                                      int &pad_top) {
  detections.clear();

  if (outputs.empty() || outputs[0].empty()) {
    return;
  }

  const std::vector<float> &output = outputs[0];
  int nc = classes_.size();
  int nm = 1;
  int total_channels = 4 + nc + nm;
  int num_detections = output.size() / total_channels;

  std::vector<types::OBB_DET> valid_detections;

  for (int i = 0; i < num_detections; i++) {
    float x = output[0 * num_detections + i];
    float y = output[1 * num_detections + i];
    float w = output[2 * num_detections + i];
    float h = output[3 * num_detections + i];

    float max_score = 0.0f;
    int best_class = 0;
    for (int c = 0; c < nc; c++) {
      float score = output[(4 + c) * num_detections + i];
      if (score > max_score) {
        max_score = score;
        best_class = c;
      }
    }

    float angle = output[(4 + nc) * num_detections + i];

    if (max_score >= conf_threshold_) {
      types::OBB_DET obj;
      obj.points = ConvertToCornerPoints(x, y, w, h, angle);
      obj.prob = max_score;
      obj.label = best_class;
      valid_detections.push_back(obj);
    }
  }

  NMS(valid_detections);

  for (auto &detection : valid_detections) {
    for (auto &point : detection.points) {
      point.x = (point.x - pad_left) * scale;
      point.y = (point.y - pad_top) * scale;
      point.x = Clamp(point.x, 0.f, (float)img.cols);
      point.y = Clamp(point.y, 0.f, (float)img.rows);
    }
  }

  detections = valid_detections;
}

void OBBDetectionYoloV11::NMS(std::vector<types::OBB_DET> &detections) {
  std::sort(detections.begin(), detections.end(),
            [](const types::OBB_DET &a, const types::OBB_DET &b) {
              return a.prob > b.prob;
            });

  std::vector<bool> keep(detections.size(), true);

  for (size_t i = 0; i < detections.size(); i++) {
    if (!keep[i])
      continue;

    float center1_x = (detections[i].points[0].x + detections[i].points[1].x +
                       detections[i].points[2].x + detections[i].points[3].x) /
                      4.0f;
    float center1_y = (detections[i].points[0].y + detections[i].points[1].y +
                       detections[i].points[2].y + detections[i].points[3].y) /
                      4.0f;

    float dx1 = detections[i].points[1].x - detections[i].points[0].x;
    float dy1 = detections[i].points[1].y - detections[i].points[0].y;
    float width1 = std::sqrt(dx1 * dx1 + dy1 * dy1);

    float dx2 = detections[i].points[3].x - detections[i].points[0].x;
    float dy2 = detections[i].points[3].y - detections[i].points[0].y;
    float height1 = std::sqrt(dx2 * dx2 + dy2 * dy2);

    float angle1 = std::atan2(dy1, dx1);

    for (size_t j = i + 1; j < detections.size(); j++) {
      if (!keep[j])
        continue;

      float center2_x =
          (detections[j].points[0].x + detections[j].points[1].x +
           detections[j].points[2].x + detections[j].points[3].x) /
          4.0f;
      float center2_y =
          (detections[j].points[0].y + detections[j].points[1].y +
           detections[j].points[2].y + detections[j].points[3].y) /
          4.0f;

      float dx1_j = detections[j].points[1].x - detections[j].points[0].x;
      float dy1_j = detections[j].points[1].y - detections[j].points[0].y;
      float width2 = std::sqrt(dx1_j * dx1_j + dy1_j * dy1_j);

      float dx2_j = detections[j].points[3].x - detections[j].points[0].x;
      float dy2_j = detections[j].points[3].y - detections[j].points[0].y;
      float height2 = std::sqrt(dx2_j * dx2_j + dy2_j * dy2_j);

      float angle2 = std::atan2(dy1_j, dx1_j);

      std::vector<float> box1 = {center1_x, center1_y, width1, height1, angle1};
      std::vector<float> box2 = {center2_x, center2_y, width2, height2, angle2};

      float iou = CalculateRotatedIoU(box1, box2);
      if (iou > iou_threshold_) {
        keep[j] = false;
      }
    }
  }

  std::vector<types::OBB_DET> filtered;
  for (size_t i = 0; i < detections.size(); i++) {
    if (keep[i]) {
      filtered.push_back(detections[i]);
    }
  }
  detections = filtered;
}

std::vector<types::Point>
OBBDetectionYoloV11::ConvertToCornerPoints(float x, float y, float w, float h,
                                           float angle) {
  std::vector<types::Point> points(4);

  float cos_a = std::cos(angle);
  float sin_a = std::sin(angle);

  float w2 = w / 2.0f;
  float h2 = h / 2.0f;

  points[0].x = x + w2 * cos_a - h2 * sin_a;
  points[0].y = y + w2 * sin_a + h2 * cos_a;
  points[1].x = x + w2 * cos_a + h2 * sin_a;
  points[1].y = y + w2 * sin_a - h2 * cos_a;
  points[2].x = x - w2 * cos_a + h2 * sin_a;
  points[2].y = y - w2 * sin_a - h2 * cos_a;
  points[3].x = x - w2 * cos_a - h2 * sin_a;
  points[3].y = y - w2 * sin_a + h2 * cos_a;

  return points;
}

std::vector<types::OBB_DET> OBBDetectionYoloV11::Detect(const cv::Mat &image) {
  std::vector<types::OBB_DET> detections;
  if (!IsInitialized()) {
    return detections;
  }

  cv::Mat resized_img;
  float scale;
  int pad_left;
  int pad_top;
  if (!Preprocess(image, resized_img, scale, pad_left, pad_top)) {
    return detections;
  }

  auto t1 = std::chrono::high_resolution_clock::now();
  std::vector<std::vector<float>> outputs = Infer(resized_img);
  auto t2 = std::chrono::high_resolution_clock::now();
  std::cout << "Inference time: " << std::chrono::duration<double, std::milli>(t2 - t1).count() << " ms" << std::endl;
  if (outputs.empty()) {
    return detections;
  }

  Postprocess(outputs, image, detections, scale, pad_left, pad_top);
  return detections;
}

cv::Mat OBBDetectionYoloV11::DrawDet(cv::Mat &image,
                                     std::vector<types::OBB_DET> &detections) {
  cv::Mat result = image.clone();

  for (auto &obj : detections) {
    std::vector<cv::Point> cv_points;
    for (const auto &point : obj.points) {
      cv_points.push_back(
          cv::Point(static_cast<int>(point.x), static_cast<int>(point.y)));
    }

    cv::polylines(result, std::vector<std::vector<cv::Point>>{cv_points}, true,
                  cv::Scalar(0, 255, 0), 2);

    std::string label =
        classes_.size() > obj.label ? classes_[obj.label] : "object";
    label += " " + std::to_string(static_cast<int>(obj.prob * 100)) + "%";

    int baseLine = 0;
    cv::Size label_size =
        cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

    float center_x = (obj.points[0].x + obj.points[1].x + obj.points[2].x +
                      obj.points[3].x) /
                     4.0f;
    float center_y = (obj.points[0].y + obj.points[1].y + obj.points[2].y +
                      obj.points[3].y) /
                     4.0f;

    cv::rectangle(
        result,
        cv::Rect(cv::Point(
                     static_cast<int>(center_x - label_size.width / 2),
                     static_cast<int>(center_y - label_size.height - baseLine)),
                 cv::Size(label_size.width, label_size.height + baseLine)),
        cv::Scalar(0, 255, 0), cv::FILLED);
    cv::putText(result, label,
                cv::Point(static_cast<int>(center_x - label_size.width / 2),
                          static_cast<int>(center_y)),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
  }

  return result;
}
