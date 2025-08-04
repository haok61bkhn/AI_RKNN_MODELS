#include "text_recognition_yolov11.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <opencv2/opencv.hpp>

TextRecognitionYoloV11::TextRecognitionYoloV11(const char *model_path,
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

TextRecognitionYoloV11::~TextRecognitionYoloV11() {}

std::vector<std::string>
TextRecognitionYoloV11::ReadClasses(const std::string &filename) {
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

float TextRecognitionYoloV11::Clamp(float val, float min, float max) {
  return val > min ? (val < max ? val : max) : min;
}

bool TextRecognitionYoloV11::Preprocess(const cv::Mat &img, cv::Mat &resized_img,
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
                     cv::BORDER_CONSTANT, cv::Scalar(255, 255, 255));
  cv::cvtColor(resized_img, resized_img, cv::COLOR_BGR2RGB);
  return true;
}

void TextRecognitionYoloV11::Postprocess(
    std::vector<std::vector<float>> &outputs, const cv::Mat &img,
    std::vector<types::TEXT_DET> &detections, float &scale, int &pad_left,
    int &pad_top) {
  detections.clear();

  if (outputs.empty() || outputs[0].empty()) {
    return;
  }

  const std::vector<float> &boxes_output = outputs[0];
  const std::vector<float> &scores_output = outputs[1];
  const std::vector<float> &classes_output = outputs[2];

  int num_detections = scores_output.size();
  std::vector<types::TEXT_DET> valid_detections;

  for (int i = 0; i < num_detections; i++) {
    float score = scores_output[i];
    if (score >= conf_threshold_) {
      types::TEXT_DET obj;
      float cx = boxes_output[i * 4 + 0];
      float cy = boxes_output[i * 4 + 1];
      float w = boxes_output[i * 4 + 2];
      float h = boxes_output[i * 4 + 3];

      obj.x1 = cx - w / 2.0f;
      obj.y1 = cy - h / 2.0f;
      obj.x2 = cx + w / 2.0f;
      obj.y2 = cy + h / 2.0f;

      obj.x1 = (obj.x1 - pad_left) * scale;
      obj.y1 = (obj.y1 - pad_top) * scale;
      obj.x2 = (obj.x2 - pad_left) * scale;
      obj.y2 = (obj.y2 - pad_top) * scale;

      obj.x1 = Clamp(obj.x1, 0.0f, (float)img.cols);
      obj.y1 = Clamp(obj.y1, 0.0f, (float)img.rows);
      obj.x2 = Clamp(obj.x2, 0.0f, (float)img.cols);
      obj.y2 = Clamp(obj.y2, 0.0f, (float)img.rows);

      if (obj.x2 <= obj.x1 || obj.y2 <= obj.y1) {
        continue;
      }

      obj.prob = score;
      obj.label = static_cast<int>(classes_output[i]);
      valid_detections.push_back(obj);
    }
  }

  NMS(valid_detections);
  detections = valid_detections;
}

void TextRecognitionYoloV11::NMS(std::vector<types::TEXT_DET> &detections) {
  std::sort(detections.begin(), detections.end(),
            [](const types::TEXT_DET &a, const types::TEXT_DET &b) {
              return a.prob > b.prob;
            });

  std::vector<bool> keep(detections.size(), true);

  for (size_t i = 0; i < detections.size(); i++) {
    if (!keep[i])
      continue;

    float area1 = (detections[i].x2 - detections[i].x1) *
                  (detections[i].y2 - detections[i].y1);

    for (size_t j = i + 1; j < detections.size(); j++) {
      if (!keep[j])
        continue;

      float area2 = (detections[j].x2 - detections[j].x1) *
                    (detections[j].y2 - detections[j].y1);

      float x1 = std::max(detections[i].x1, detections[j].x1);
      float y1 = std::max(detections[i].y1, detections[j].y1);
      float x2 = std::min(detections[i].x2, detections[j].x2);
      float y2 = std::min(detections[i].y2, detections[j].y2);

      if (x2 > x1 && y2 > y1) {
        float intersection = (x2 - x1) * (y2 - y1);
        float union_area = area1 + area2 - intersection;
        float iou = intersection / union_area;

        if (iou > iou_threshold_) {
          keep[j] = false;
        }
      }
    }
  }

  std::vector<types::TEXT_DET> filtered;
  for (size_t i = 0; i < detections.size(); i++) {
    if (keep[i]) {
      filtered.push_back(detections[i]);
    }
  }
  detections = filtered;
}

std::vector<types::TEXT_DET>
TextRecognitionYoloV11::Detect(const cv::Mat &image) {
  std::vector<types::TEXT_DET> detections;
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
  std::cout << "Inference time: "
            << std::chrono::duration<double, std::milli>(t2 - t1).count()
            << " ms" << std::endl;
  if (outputs.empty()) {
    return detections;
  }

  Postprocess(outputs, image, detections, scale, pad_left, pad_top);
  return detections;
}

std::pair<std::string, std::vector<float>>
TextRecognitionYoloV11::Predict(const cv::Mat &image) {
  std::vector<types::TEXT_DET> detections = Detect(image);
  if (detections.empty()) {
    return std::make_pair("", std::vector<float>());
  }

  std::sort(detections.begin(), detections.end(),
            [](const types::TEXT_DET &a, const types::TEXT_DET &b) {
              return a.x1 < b.x1;
            });

  std::string text = "";
  std::vector<float> confidences;

  for (const auto &detection : detections) {
    std::string class_name =
        detection.label < (int)classes_.size()
            ? classes_[detection.label]
            : std::to_string(detection.label);
    text += class_name;
    confidences.push_back(detection.prob);
  }

  return std::make_pair(text, confidences);
}

cv::Mat
TextRecognitionYoloV11::DrawDet(cv::Mat &image,
                                std::vector<types::TEXT_DET> &detections) {
  cv::Mat result = image.clone();

  for (auto &obj : detections) {
    cv::rectangle(result, cv::Point(static_cast<int>(obj.x1), static_cast<int>(obj.y1)),
                  cv::Point(static_cast<int>(obj.x2), static_cast<int>(obj.y2)),
                  cv::Scalar(0, 255, 0), 2);

    std::string class_name = obj.label < (int)classes_.size()
                                 ? classes_[obj.label]
                                 : std::to_string(obj.label);
    std::string label = class_name + ": " + std::to_string(static_cast<int>(obj.prob * 100)) + "%";

    int baseLine = 0;
    cv::Size label_size =
        cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

    cv::rectangle(result,
                  cv::Point(static_cast<int>(obj.x1),
                            static_cast<int>(obj.y1) - label_size.height - 4),
                  cv::Point(static_cast<int>(obj.x1) + label_size.width,
                            static_cast<int>(obj.y1)),
                  cv::Scalar(0, 255, 0), cv::FILLED);
    cv::putText(result, label,
                cv::Point(static_cast<int>(obj.x1), static_cast<int>(obj.y1) - 2),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
  }

  return result;
} 