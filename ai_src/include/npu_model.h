#ifndef NPU_MODEL_H
#define NPU_MODEL_H

#include "rknn_api.h"
#include <opencv2/opencv.hpp>
#include <vector>

struct rknn_app_context {
    rknn_context rknn_ctx;
    rknn_input_output_num io_num;
    rknn_tensor_attr* input_attrs;
    rknn_tensor_attr* output_attrs;
};

class NPUModel {
public:
    NPUModel(const char* model_path);
    virtual ~NPUModel();
    
    bool IsInitialized() const;
    virtual bool Preprocess(const cv::Mat& img, cv::Mat& resized_img, float& scale);
    std::vector<std::vector<float>> Infer(const cv::Mat& img);

protected:
    bool initialized_;
    int input_channel_;
    int input_height_;
    int input_width_;
    rknn_context rknn_ctx_;
    rknn_app_context rknn_app_ctx_;
};

#endif // NPU_MODEL_H 