#include "npu_model.h"
#include <cstring>
#include <fstream>

static unsigned char* load_model(const char* filename, int& model_size) {
    FILE* fp = fopen(filename, "rb");
    if (fp == nullptr) {
        return nullptr;
    }

    fseek(fp, 0, SEEK_END);
    model_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    unsigned char* model_data = (unsigned char*)malloc(model_size);
    if (model_data == nullptr) {
        fclose(fp);
        return nullptr;
    }

    size_t read_size = fread(model_data, 1, model_size, fp);
    if (read_size != static_cast<size_t>(model_size)) {
        free(model_data);
        fclose(fp);
        return nullptr;
    }

    fclose(fp);
    return model_data;
}

NPUModel::NPUModel(const char* model_path) : initialized_(false) {
    int model_data_size = 0;
    unsigned char* model_data = load_model(model_path, model_data_size);
    if (!model_data) return;
    int ret = rknn_init(&rknn_app_ctx_.rknn_ctx, model_data, model_data_size, 0, NULL);
    free(model_data);
    if (ret < 0) return;
    ret = rknn_set_core_mask(rknn_app_ctx_.rknn_ctx, RKNN_NPU_CORE_AUTO);
    if (ret < 0) return;
    rknn_input_output_num io_num;
    ret = rknn_query(rknn_app_ctx_.rknn_ctx, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC) return;
    rknn_app_ctx_.io_num = io_num;
    rknn_tensor_attr input_attrs[io_num.n_input];
    memset(input_attrs, 0, sizeof(input_attrs));
    for (uint32_t i = 0; i < io_num.n_input; i++) {
        input_attrs[i].index = i;
        ret = rknn_query(rknn_app_ctx_.rknn_ctx, RKNN_QUERY_INPUT_ATTR, &(input_attrs[i]), sizeof(rknn_tensor_attr));
        if (ret < 0) return;
    }
    if (input_attrs[0].fmt == RKNN_TENSOR_NCHW) {
        input_channel_ = input_attrs[0].dims[1];
        input_height_ = input_attrs[0].dims[2];
        input_width_ = input_attrs[0].dims[3];
    } else {
        input_height_ = input_attrs[0].dims[1];
        input_width_ = input_attrs[0].dims[2];
        input_channel_ = input_attrs[0].dims[3];
    }
    rknn_app_ctx_.input_attrs = (rknn_tensor_attr*)malloc(io_num.n_input * sizeof(rknn_tensor_attr));
    rknn_app_ctx_.output_attrs = (rknn_tensor_attr*)malloc(io_num.n_output * sizeof(rknn_tensor_attr));
    if (!rknn_app_ctx_.input_attrs || !rknn_app_ctx_.output_attrs) return;
    memcpy(rknn_app_ctx_.input_attrs, input_attrs, io_num.n_input * sizeof(rknn_tensor_attr));
    rknn_tensor_attr output_attrs[io_num.n_output];
    memset(output_attrs, 0, sizeof(output_attrs));
    for (uint32_t i = 0; i < io_num.n_output; i++) {
        output_attrs[i].index = i;
        ret = rknn_query(rknn_app_ctx_.rknn_ctx, RKNN_QUERY_OUTPUT_ATTR, &(output_attrs[i]), sizeof(rknn_tensor_attr));
    }
    memcpy(rknn_app_ctx_.output_attrs, output_attrs, io_num.n_output * sizeof(rknn_tensor_attr));
    initialized_ = true;
}

NPUModel::~NPUModel() {
    if (initialized_) rknn_destroy(rknn_app_ctx_.rknn_ctx);
    if (rknn_app_ctx_.input_attrs) free(rknn_app_ctx_.input_attrs);
    if (rknn_app_ctx_.output_attrs) free(rknn_app_ctx_.output_attrs);
}

bool NPUModel::IsInitialized() const {
    return initialized_;
}

bool NPUModel::Preprocess(const cv::Mat& img, cv::Mat& resized_img, float& scale) {
    if (img.cols == input_width_ && img.rows == input_height_) {
        resized_img = img.clone();
        scale = 1.0f;
        return true;
    }
    cv::resize(img, resized_img, cv::Size(input_width_, input_height_));
    scale = static_cast<float>(input_width_) / img.cols;
    return true;
}

std::vector<std::vector<float>> NPUModel::Infer(const cv::Mat& img) {
    std::vector<std::vector<float>> result;
    if (!initialized_) return result;
    cv::Mat resized_img;
    float scale;
    if (!Preprocess(img, resized_img, scale)) return result;
    rknn_input inputs[1];
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].size = input_width_ * input_height_ * input_channel_;
    inputs[0].buf = (void*)resized_img.data;
    int ret = rknn_inputs_set(rknn_app_ctx_.rknn_ctx, rknn_app_ctx_.io_num.n_input, inputs);
    if (ret < 0) return result;
    
    std::vector<rknn_output> outputs(rknn_app_ctx_.io_num.n_output);
    for (uint32_t i = 0; i < rknn_app_ctx_.io_num.n_output; i++) {
        outputs[i].index = i;
        outputs[i].want_float = true;
    }
    
    ret = rknn_run(rknn_app_ctx_.rknn_ctx, nullptr);
    if (ret < 0) return result;
    ret = rknn_outputs_get(rknn_app_ctx_.rknn_ctx, rknn_app_ctx_.io_num.n_output, outputs.data(), nullptr);
    if (ret < 0) return result;
    for (uint32_t i = 0; i < rknn_app_ctx_.io_num.n_output; i++) {
        float* out_ptr = (float*)outputs[i].buf;
        int out_size = outputs[i].size / sizeof(float);
        std::vector<float> out_data(out_ptr, out_ptr + out_size);
        result.push_back(out_data);
    }
    rknn_outputs_release(rknn_app_ctx_.rknn_ctx, rknn_app_ctx_.io_num.n_output, outputs.data());
    return result;
} 