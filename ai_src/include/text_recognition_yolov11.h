#ifndef TEXT_RECOGNITION_YOLOV11_H
#define TEXT_RECOGNITION_YOLOV11_H

#include "npu_model.h"
#include "types.h"
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

class TextRecognitionYoloV11 : public NPUModel {
public:
  TextRecognitionYoloV11(const char *model_path, const char *classes_file,
                         float conf_threshold = 0.5f, float iou_threshold = 0.5f);
  ~TextRecognitionYoloV11();

  std::vector<types::TEXT_DET> Detect(const cv::Mat &image);
  std::pair<std::string, std::vector<float>> Predict(const cv::Mat &image);
  cv::Mat DrawDet(cv::Mat &image, std::vector<types::TEXT_DET> &detections);

private:
  std::vector<std::string> ReadClasses(const std::string &filename);
  bool Preprocess(const cv::Mat &img, cv::Mat &resized_img, float &scale,
                  int &pad_left, int &pad_top);
  void Postprocess(std::vector<std::vector<float>> &outputs, const cv::Mat &img,
                   std::vector<types::TEXT_DET> &detections, float &scale,
                   int &pad_left, int &pad_top);
  void NMS(std::vector<types::TEXT_DET> &detections);
  float Clamp(float val, float min, float max);

  std::vector<std::string> classes_;
  float conf_threshold_;
  float iou_threshold_;
};

#endif 