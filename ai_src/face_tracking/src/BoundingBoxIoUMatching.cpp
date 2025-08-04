#include "BoundingBoxIoUMatching.h"

Eigen::MatrixXd box_iou_batch(const Eigen::MatrixXd &track_boxes,
                              const Eigen::MatrixXd &detection_boxes) {
  if (track_boxes.cols() != 4 || detection_boxes.cols() != 4) {
    throw std::invalid_argument("Input matrices must have 4 columns each.");
  }
  int N = (int)track_boxes.rows();
  int M = (int)detection_boxes.rows();
  Eigen::VectorXd track_areas =
      (track_boxes.col(2) - track_boxes.col(0))
          .cwiseProduct(track_boxes.col(3) - track_boxes.col(1));
  Eigen::VectorXd detection_areas =
      (detection_boxes.col(2) - detection_boxes.col(0))
          .cwiseProduct(detection_boxes.col(3) - detection_boxes.col(1));
  Eigen::MatrixXd iou_matrix(N, M);
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < M; ++j) {
      double inter_x_min = std::max(track_boxes(i, 0), detection_boxes(j, 0));
      double inter_y_min = std::max(track_boxes(i, 1), detection_boxes(j, 1));
      double inter_x_max = std::min(track_boxes(i, 2), detection_boxes(j, 2));
      double inter_y_max = std::min(track_boxes(i, 3), detection_boxes(j, 3));
      double inter_width = std::max(inter_x_max - inter_x_min, 0.0);
      double inter_height = std::max(inter_y_max - inter_y_min, 0.0);
      double inter_area = inter_width * inter_height;
      iou_matrix(i, j) =
          inter_area / (track_areas(i) + detection_areas(j) - inter_area);
    }
  }
  return iou_matrix;
}

Eigen::MatrixXd iou_distance(const std::vector<KalmanBBoxTrack> &track_list_a,
                             const std::vector<KalmanBBoxTrack> &track_list_b) {
  size_t m = track_list_a.size();
  size_t n = track_list_b.size();
  Eigen::MatrixXd tlbr_list_a(m, 4);
  for (size_t i = 0; i < m; i++) {
    tlbr_list_a.row(i) = track_list_a[i].tlbr();
  }

  Eigen::MatrixXd tlbr_list_b(n, 4);
  for (size_t i = 0; i < n; i++) {
    tlbr_list_b.row(i) = track_list_b[i].tlbr();
  }
  Eigen::MatrixXd ious;
  if (tlbr_list_a.rows() == 0 || tlbr_list_b.rows() == 0) {
    ious = Eigen::MatrixXd::Zero(tlbr_list_a.rows(), tlbr_list_b.rows());
  } else {
    ious = box_iou_batch(tlbr_list_a, tlbr_list_b);
  }
  Eigen::MatrixXd cost_matrix = Eigen::MatrixXd::Ones(m, n) - ious;

  return cost_matrix;
}

std::vector<int>
match_detections_with_tracks(const Eigen::MatrixXd &tlbr_boxes,
                             const std::vector<KalmanBBoxTrack> &tracks) {
  size_t m = tracks.size();
  size_t n = tlbr_boxes.rows();
  std::vector<int> track_ids(n, -1);
  Eigen::MatrixXd track_boxes(m, 4);
  for (size_t i = 0; i < m; i++) {
    track_boxes.row(i) = tracks[i].tlbr();
  }
  Eigen::MatrixXd iou = box_iou_batch(track_boxes, tlbr_boxes);
  for (size_t i = 0; i < m; i++) {
    int idx_max = -1;
    double max_val = 0;
    for (size_t j = 0; j < n; j++) {
      if (iou(i, j) > max_val) {
        max_val = iou(i, j);
        idx_max = (int)j;
      }
    }
    if (max_val > 0) {
      track_ids[idx_max] = tracks[i].get_track_id();
    }
  }
  return track_ids;
}
