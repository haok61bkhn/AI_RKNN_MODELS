#ifndef OBB_DETECTION_YOLOV11_H_
#define OBB_DETECTION_YOLOV11_H_

#include "npu_model.h"
#include "types.h"
#include <opencv2/opencv.hpp>
#include <vector>

class OBBDetectionYoloV11 : public NPUModel {
public:
  OBBDetectionYoloV11(const char *model_path, const char *classes_file,
                      float conf_threshold = 0.25, float iou_threshold = 0.45);
  ~OBBDetectionYoloV11();

  std::vector<types::OBB_DET> Detect(const cv::Mat &image);
  cv::Mat DrawDet(cv::Mat &image, std::vector<types::OBB_DET> &detections);

private:
  std::vector<std::string> ReadClasses(const std::string &filename);
  bool Preprocess(const cv::Mat &img, cv::Mat &resized_img, float &scale,
                  int &pad_left, int &pad_top);
  void Postprocess(std::vector<std::vector<float>> &outputs, const cv::Mat &img,
                   std::vector<types::OBB_DET> &detections, float &scale,
                   int &pad_left, int &pad_top);
  float Clamp(float val, float min, float max);
  float CalculateRotatedIoU(const std::vector<float> &box1,
                            const std::vector<float> &box2);
  std::vector<types::Point> ConvertToCornerPoints(float x, float y, float w,
                                                  float h, float angle);
  void NMS(std::vector<types::OBB_DET> &detections);

private:
  std::vector<std::string> classes_;
  float conf_threshold_;
  float iou_threshold_;
};

#endif