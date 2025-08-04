#include "text_recognition_yolov11.h"
#include <fstream>
#include <chrono>

int main() {
  std::string model_path = "models/text_recognition_yolov11_quantized.rknn";
  std::string classes_file = "models/text_recognition_yolov11_classes.txt";
  TextRecognitionYoloV11 text_recognition(model_path.c_str(), classes_file.c_str());

  std::ifstream infile("models/text_crop.txt");
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
    std::vector<types::TEXT_DET> detections = text_recognition.Detect(image);
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
    total_time += elapsed;
    count++;
    
    auto prediction = text_recognition.Predict(image);
    std::string recognized_text = prediction.first;
    std::vector<float> confidences = prediction.second;
    
    std::cout << "Image: " << image_path << ", detections: " << detections.size() 
              << ", recognized text: '" << recognized_text << "', time: " << elapsed << " ms" << std::endl;
    
    if (!detections.empty()) {
      cv::Mat result = text_recognition.DrawDet(image, detections);
      std::string out_path = "results/text_result_" + std::to_string(count) + ".jpg";
      cv::imwrite(out_path, result);
    }
  }
  if (count > 0) {
    std::cout << "Average detection time: " << (total_time / count) << " ms" << std::endl;
  }
  return 0;
} 