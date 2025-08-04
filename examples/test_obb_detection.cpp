#include "obb_detection_yolov11.h"
#include <fstream>
#include <chrono>

int main() {
  std::string model_path = "models/text_obb_quantized.rknn";
  std::string classes_file = "models/obb_classes.txt";
  OBBDetectionYoloV11 obb_detection(model_path.c_str(), classes_file.c_str());

  std::ifstream infile("models/obb_datasets.txt");
  std::vector<std::string> image_paths;
  std::string line;
  while (std::getline(infile, line)) {
    if (!line.empty()) {
      image_paths.push_back("models/" + line);
    }
  }

  double total_time = 0.0;
  int count = 0;
  for (const auto& image_path : image_paths) {
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
      continue;
    }
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<types::OBB_DET> detections = obb_detection.Detect(image);
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
    total_time += elapsed;
    count++;
    std::cout << "Image: " << image_path << ", detections: " << detections.size() << ", time: " << elapsed << " ms" << std::endl;
    cv::Mat result = obb_detection.DrawDet(image, detections);
    std::string out_path = "results/obb_result_" + std::to_string(count) + ".jpg";
    cv::imwrite(out_path, result);
  }
  if (count > 0) {
    std::cout << "Average detection time: " << (total_time / count) << " ms" << std::endl;
  }
  return 0;
}