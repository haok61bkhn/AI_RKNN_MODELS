#include "face_detection.h"

FaceDetector::FaceDetector(const char *model_path, float obj_threshold,
                           float nms_threshold, bool using_tracking)
    : NPUModel(model_path), obj_threshold_(obj_threshold),
      nms_threshold_(nms_threshold), using_tracking_(using_tracking) {
  if (!IsInitialized()) {
    return;
  }
  face_aligner = FaceAligner();
  head_pose_estimator = HeadPoseEstimator();
  if (using_tracking_) {
    face_tracker = new BYTETracker();
  }
  list_stride_ = {8, 16, 32};
  for (auto &stride : list_stride_) {
    std::vector<std::vector<float>> anchor_centers =
        GenerateAnchorCenters(stride);
    all_anchor_centers_.emplace_back(anchor_centers);
  }
}

FaceDetector::~FaceDetector() {}

std::vector<std::vector<float>>
FaceDetector::GenerateAnchorCenters(int stride) {
  int input_height = input_height_;
  int input_width = input_width_;
  assert(input_width % 32 == 0 && input_height % 32 == 0);

  std::vector<std::vector<float>> anchors;
  int size_height = input_height / stride;
  int size_width = input_width / stride;
  float cy = 0;
  for (int i = 0; i < size_height; i++) {
    float cx = 0;
    for (int k = 0; k < size_width; k++) {
      std::vector<float> anchor{cx, cy};
      anchors.emplace_back(anchor);
      anchors.emplace_back(anchor);
      cx = cx + (float)stride;
    }
    cy = cy + (float)stride;
  }
  return anchors;
}

void FaceDetector::GenerateProposals(
    const std::vector<std::vector<float>> &anchors, int feat_stride,
    const std::vector<float> &score_blob, const std::vector<float> &bbox_blob,
    const std::vector<float> &kps_blob,
    std::vector<types::FaceDetectRes> &face_objects) {
  for (size_t i = 0; i < score_blob.size(); i++) {
    float prob = score_blob[i];
    if (prob >= obj_threshold_) {
      if (i * 4 + 3 >= bbox_blob.size()) {
        continue;
      }

      float box0 = bbox_blob[i * 4 + 0];
      float box1 = bbox_blob[i * 4 + 1];
      float box2 = bbox_blob[i * 4 + 2];
      float box3 = bbox_blob[i * 4 + 3];

      if (i >= anchors.size()) {
        continue;
      }

      float cx = anchors[i][0];
      float cy = anchors[i][1];

      float x1 = cx - box0 * feat_stride;
      float y1 = cy - box1 * feat_stride;
      float x2 = cx + box2 * feat_stride;
      float y2 = cy + box3 * feat_stride;

      if (x1 < 0)
        x1 = 0;
      if (y1 < 0)
        y1 = 0;
      if (x2 > input_width_)
        x2 = input_width_;
      if (y2 > input_height_)
        y2 = input_height_;
      if (x2 <= x1 || y2 <= y1) {
        continue;
      }

      types::FaceDetectRes box;
      box.UpdateBox(x1, y1, x2, y2, prob);

      if (i * 10 + 9 >= kps_blob.size()) {
        continue;
      }

      for (int k = 0; k < 5; k++) {
        box.landmark[k].x = cx + kps_blob[i * 10 + k * 2] * feat_stride;
        box.landmark[k].y = cy + kps_blob[i * 10 + k * 2 + 1] * feat_stride;
      }
      face_objects.push_back(box);
    }
  }
}

void FaceDetector::NMS(std::vector<types::FaceDetectRes> &input_boxes) {
  std::vector<types::FaceDetectRes> keep;
  std::vector<float> varea(input_boxes.size());
  for (size_t i = 0; i < input_boxes.size(); i++) {
    varea[i] = (input_boxes[i].x2 - input_boxes[i].x1 + 1) *
               (input_boxes[i].y2 - input_boxes[i].y1 + 1);
  }

  std::vector<int> indices(input_boxes.size());
  std::iota(indices.begin(), indices.end(), 0);

  while (indices.size() > 0) {
    int idx = indices[0];
    keep.push_back(input_boxes[idx]);

    std::vector<int> tmp_indices;
    for (size_t i = 1; i < indices.size(); i++) {
      int idx2 = indices[i];
      float xx1 = std::max(input_boxes[idx].x1, input_boxes[idx2].x1);
      float yy1 = std::max(input_boxes[idx].y1, input_boxes[idx2].y1);
      float xx2 = std::min(input_boxes[idx].x2, input_boxes[idx2].x2);
      float yy2 = std::min(input_boxes[idx].y2, input_boxes[idx2].y2);

      float w = std::max(0.0f, xx2 - xx1 + 1);
      float h = std::max(0.0f, yy2 - yy1 + 1);
      float inter = w * h;
      float ovr = inter / (varea[idx] + varea[idx2] - inter);

      if (ovr < nms_threshold_) {
        tmp_indices.push_back(idx2);
      }
    }
    indices = tmp_indices;
  }

  input_boxes = keep;
}

void FaceDetector::FilterBoxes(
    std::vector<types::FaceDetectRes> &predicted_boxes) {
  std::sort(predicted_boxes.begin(), predicted_boxes.end(),
            [](const types::FaceDetectRes &a, const types::FaceDetectRes &b) {
              return a.score > b.score;
            });
  NMS(predicted_boxes);
}

bool FaceDetector::Preprocess(const cv::Mat &img, cv::Mat &resized_img,
                              float &scale) {
  int long_side = std::max(img.cols, img.rows);
  int kmodel_inputsize = std::max(input_height_, input_width_);

  if (long_side <= kmodel_inputsize) {
    scale = 1.0;
    resized_img = img.clone();
  } else {
    scale = std::min(float(input_width_) / float(img.cols),
                     float(input_height_) / float(img.rows));
    cv::resize(img, resized_img, cv::Size(), scale, scale, cv::INTER_LINEAR);
  }

  int pad_width_size = input_width_ - resized_img.cols;
  int pad_height_size = input_height_ - resized_img.rows;
  cv::copyMakeBorder(resized_img, resized_img, 0, pad_height_size, 0,
                     pad_width_size, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
  cv::cvtColor(resized_img, resized_img, cv::COLOR_BGR2RGB);
  return true;
}

void FaceDetector::Postprocess(
    std::vector<std::vector<float>> &outputs, const cv::Mat &img,
    std::vector<types::FaceDetectRes> &face_detections, float &scale) {
  for (size_t i = 0; i < list_stride_.size(); i++) {
    if (i * 3 + 2 >= rknn_app_ctx_.io_num.n_output) {
      break;
    }
    std::vector<types::FaceDetectRes> face_objects;
    auto score_blob = outputs[i];
    auto bbox_blob = outputs[3 + i];
    auto kps_blob = outputs[6 + i];
    GenerateProposals(all_anchor_centers_[i], list_stride_[i], score_blob,
                      bbox_blob, kps_blob, face_objects);

    face_detections.insert(face_detections.end(), face_objects.begin(),
                           face_objects.end());
  }

  FilterBoxes(face_detections);
  for (auto &box : face_detections) {
    box.x1 = box.x1 / scale;
    box.y1 = box.y1 / scale;
    box.x2 = box.x2 / scale;
    box.y2 = box.y2 / scale;

    box.x1 = std::max(0.0f, std::min(box.x1, static_cast<float>(img.cols - 1)));
    box.y1 = std::max(0.0f, std::min(box.y1, static_cast<float>(img.rows - 1)));
    box.x2 = std::max(0.0f, std::min(box.x2, static_cast<float>(img.cols - 1)));
    box.y2 = std::max(0.0f, std::min(box.y2, static_cast<float>(img.rows - 1)));
    for (int k = 0; k < 5; k++) {
      box.landmark[k].x = box.landmark[k].x / scale;
      box.landmark[k].y = box.landmark[k].y / scale;

      box.landmark[k].x = std::max(
          0.0f, std::min(box.landmark[k].x, static_cast<float>(img.cols - 1)));
      box.landmark[k].y = std::max(
          0.0f, std::min(box.landmark[k].y, static_cast<float>(img.rows - 1)));
    }
  }
}

std::vector<types::FaceDetectRes> FaceDetector::Detect(const cv::Mat &img) {
  std::vector<types::FaceDetectRes> face_detections;
  if (!IsInitialized())
    return face_detections;
  cv::Mat resized_img;
  float scale;
  if (!Preprocess(img, resized_img, scale))
    return face_detections;
  std::vector<std::vector<float>> outputs = Infer(resized_img);
  if (outputs.empty())
    return face_detections;
  Postprocess(outputs, img, face_detections, scale);
  if (using_tracking_) {
    face_tracker->TrackFace(face_detections);
  }
  return face_detections;
}

void FaceDetector::AlignFace(const cv::Mat &image,
                             types::FaceDetectRes &face_detect_res) {
  face_detect_res.aligned_face =
      face_aligner.AlignFace(image, face_detect_res.landmark);
}

void FaceDetector::SetCameraSize(int camera_width, int camera_height) {
  head_pose_estimator.SetCameraSize(camera_width, camera_height);
}

void FaceDetector::EstimateHeadPose(types::FaceDetectRes &face_detect_res) {
  face_detect_res.orientation =
      head_pose_estimator.EstimateHeadPose(face_detect_res.landmark);
}

void FaceDetector::DrawOutput(cv::Mat &img,
                              std::vector<types::FaceDetectRes> &boxes,
                              bool landmark) {
  for (int j = 0; j < boxes.size(); ++j) {
    // if (!boxes[j].is_good_face)
    //   continue;
    cv::Rect rect(boxes[j].x1, boxes[j].y1, boxes[j].x2 - boxes[j].x1,
                  boxes[j].y2 - boxes[j].y1);
    cv::rectangle(img, rect, cv::Scalar(0, 255, 0), 1, 8, 0);
    char test[80];
    int score = int(boxes[j].score * 100);
    if (using_tracking_) {
      sprintf(test, "%d %d", boxes[j].id_tracking, score);
    } else {
      sprintf(test, "%d", score);
    }

    // cv::putText(img, test, cv::Size((boxes[j].x1), boxes[j].y1),
    //             cv::FONT_HERSHEY_COMPLEX, 0.8, cv::Scalar(0, 255, 0));

    // if (landmark) {

    //   cv::circle(img, cv::Point(boxes[j].landmark[0].x,
    //   boxes[j].landmark[0].y),
    //              1, cv::Scalar(0, 0, 225), 1);
    //   cv::circle(img, cv::Point(boxes[j].landmark[1].x,
    //   boxes[j].landmark[1].y),
    //              1, cv::Scalar(0, 255, 225), 1);
    //   cv::circle(img, cv::Point(boxes[j].landmark[2].x,
    //   boxes[j].landmark[2].y),
    //              1, cv::Scalar(255, 0, 225), 1);
    //   cv::circle(img, cv::Point(boxes[j].landmark[3].x,
    //   boxes[j].landmark[3].y),
    //              1, cv::Scalar(0, 255, 0), 1);
    //   cv::circle(img, cv::Point(boxes[j].landmark[4].x,
    //   boxes[j].landmark[4].y),
    //              1, cv::Scalar(255, 0, 0), 1);
    // }
  }
}