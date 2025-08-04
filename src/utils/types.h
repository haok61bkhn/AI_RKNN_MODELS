#ifndef TYPES_H_
#define TYPES_H_
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace types {

class FaceInfo {
public:
  FaceInfo() {}
  FaceInfo(const std::string &face_id, const std::string &face_name,
           const std::string &face_image_path)
      : face_id(face_id), face_name(face_name),
        face_image_path(face_image_path) {}

  void SetFaceInfo(const types::FaceInfo &face_info) {
    this->face_id = face_info.face_id;
    this->face_name = face_info.face_name;
    this->face_image_path = face_info.face_image_path;
  }

public:
  std::string face_id;
  std::string face_name;
  std::string face_image_path;
};

class Result {
public:
  std::string label;
  float confidence;
  cv::Mat cropped_image;
  std::vector<float> feature;
  std::vector<int> box = {0, 0, 0, 0};
};
struct Point {
  float x;
  float y;
  float prob;
};
struct Orientation {
  float yaw;
  float pitch;
  float roll;
};

struct FaceDetectRes {

  void UpdateBox(float x1, float y1, float x2, float y2, float score) {
    this->x1 = x1;
    this->y1 = y1;
    this->x2 = x2;
    this->y2 = y2;
    this->score = score;
    this->width = x2 - x1;
    this->height = y2 - y1;
  }

  float x1;
  float y1;
  float x2;
  float y2;
  int width;
  int height;
  std::vector<Point> landmark = std::vector<Point>(5);
  cv::Mat aligned_face;
  cv::Mat cropped_face;

  Orientation orientation;
  float ats_confidence;
  bool is_good_face = false;
  bool is_real_face = false;
  float score;
  int id_tracking;

  int timestamp;
  std::string camera_id;
};

struct DetectionFrame {
  DetectionFrame() {}
  DetectionFrame(std::vector<FaceDetectRes> &detections, cv::Mat &frame,
                 int timestamp)
      : detections(detections), frame(frame), timestamp(timestamp) {}
  std::vector<FaceDetectRes> detections;
  cv::Mat frame;
  int timestamp;
};

struct FaceSearchResult {
  std::string pid;
  float min_distance;
  float score;
  void SetFaceInfo(const types::FaceInfo &face_info) {
    this->face_info.SetFaceInfo(face_info);
  }
  types::FaceInfo face_info;
};

struct FaceCheckerModelConfig {
  float scale;
  float shift_x;
  float shift_y;
  int height;
  int width;
  bool org_resize;
};

struct OBB_DET {
  std::vector<Point> points;
  float prob;
  int label;
};

struct TEXT_DET {
  float x1, y1, x2, y2;
  float prob;
  int label;
};
} // namespace types

#endif