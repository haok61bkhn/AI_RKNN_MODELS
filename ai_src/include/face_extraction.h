#ifndef FACE_EXTRACTION_H
#define FACE_EXTRACTION_H
#include <opencv2/opencv.hpp>
#include "npu_model.h"
#include "types.h"

class FaceExtractor : public NPUModel {
 public:
  FaceExtractor(const char *model_path);
  ~FaceExtractor();
  void ExtractFace(const cv::Mat& img, std::vector<float>& feature);

 private:
  float Norm(std::vector<float> const& u);
  void L2Norm(std::vector<float>& v);
  int batch_size_;
};
#endif