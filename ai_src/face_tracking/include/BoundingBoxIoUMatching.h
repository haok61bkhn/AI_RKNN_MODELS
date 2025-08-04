#pragma once

#include "Export.h"
#include "KalmanBBoxTrack.h"
#include <Eigen/Dense>
#include <iostream>
#include <vector>

Eigen::MatrixXd box_iou_batch(const Eigen::MatrixXd &track_boxes,
                              const Eigen::MatrixXd &detection_boxes);

Eigen::MatrixXd iou_distance(const std::vector<KalmanBBoxTrack> &track_list_a,
                             const std::vector<KalmanBBoxTrack> &track_list_b);

BYTE_TRACK_EIGEN_API std::vector<int>
match_detections_with_tracks(const Eigen::MatrixXd &tlbr_boxes,
                             const std::vector<KalmanBBoxTrack> &tracks);
