#ifndef MODELS_FACE_LIVE_CHECKER_H_
#define MODELS_FACE_LIVE_CHECKER_H_
#include "npu_model.h"
#include "types.h"
#include <opencv2/opencv.hpp>

class FaceAntispoofingCheckerModel : public NPUModel {
public:
  FaceAntispoofingCheckerModel(const char *model_path, const float &scale,
                               const int &batch_size = 1,
                               const bool &org_resize = false,
                               const float &shift_x = 0.0,
                               const float &shift_y = 0.0,
                               const int &height = 80, const int &width = 80);
  ~FaceAntispoofingCheckerModel();

  void Predict(const cv::Mat &img, const cv::Rect &face_det,
               std::vector<float> &pred);
  void Preprocess(const cv::Mat &img, const cv::Rect &box,
                  cv::Mat &processed_img);

private:
  std::vector<float> Softmax(const std::vector<float> &input);
  cv::Rect CalculateBox(const cv::Rect &box, int src_w, int src_h,
                        types::FaceCheckerModelConfig &config);

private:
  types::FaceCheckerModelConfig config_;
};

class FaceAntispoofingChecker {
public:
  FaceAntispoofingChecker(const std::vector<std::string> &model_paths,
                          const float &threshold);
  ~FaceAntispoofingChecker();
  bool Predict(const cv::Mat &img, const cv::Rect &face_det);
  void DrawFaceChecker(cv::Mat &img, std::vector<types::FaceDetectRes> &faces);

private:
  int num_models_ = 2;
  std::vector<FaceAntispoofingCheckerModel *> models_;
  float threshold_ = 0.95;
  std::vector<float> scales_ = {2.7, 4.0};
};

#endif