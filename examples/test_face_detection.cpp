#include "face_detection.h"

int main(int argc, char *argv[]) {
  std::string model_path = "models/scrfd_500m_320_1_quantized.rknn";
  bool using_tracking = false;
  float obj_threshold = 0.3;
  float nms_threshold = 0.5;
  FaceDetector face_detector(model_path.c_str(), obj_threshold, nms_threshold,
                             using_tracking);
  cv::Mat img = cv::imread("image_examples/st.jpg");
  std::vector<types::FaceDetectRes> face_detections = face_detector.Detect(img);
  face_detector.DrawOutput(img, face_detections, true);
  cv::imwrite("results/face_detection_output.jpg", img);
  std::cout << "face_detections.size(): " << face_detections.size()
            << std::endl;
  return 0;
}