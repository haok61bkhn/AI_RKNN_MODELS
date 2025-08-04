#include "face_extraction.h"

FaceExtractor::FaceExtractor(const char *model_path) : NPUModel(model_path) {
  if (!IsInitialized()) {
    return;
  }
}

FaceExtractor::~FaceExtractor() {}

float FaceExtractor::Norm(std::vector<float> const &u) {
  float accum = 0.;
  for (int i = 0; i < u.size(); ++i) {
    accum += u[i] * u[i];
  }
  return std::sqrt(accum);
}

void FaceExtractor::L2Norm(std::vector<float> &v) {
  float k = Norm(v);
  std::transform(v.begin(), v.end(), v.begin(),
                 [k](float &c) { return c / k; });
}

void FaceExtractor::ExtractFace(const cv::Mat &img,
                                std::vector<float> &feature) {
  cv::Mat rgb_img;
  cv::cvtColor(img, rgb_img, cv::COLOR_BGR2RGB);
  feature = Infer(rgb_img)[0];
  L2Norm(feature);
}