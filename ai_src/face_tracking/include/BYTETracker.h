#pragma once

#include "BaseTrack.h"
#include "BoundingBoxIoUMatching.h"
#include "BoundingBoxTrackUtils.h"
#include "Export.h"
#include "KalmanBBoxTrack.h"
#include "KalmanFilter.h"
#include "LinearAssignment.h"
#include "types.h"
#include <memory>
#include <sstream>
#include <stdexcept>
#include <stdlib.h>
#include <vector>

class BYTE_TRACK_EIGEN_API BYTETracker {
public:
  BYTETracker(float track_thresh = 0.25, int track_buffer = 30,
              float match_thresh = 0.8, int frame_rate = 30);

private:
  const float BASE_FRAME_RATE = 30.0;
  const float MIN_KEEP_THRESH = 0.1f;
  const float LOWER_CONFIDENCE_MATCHING_THRESHOLD = 0.5;
  const float ACTIVATION_MATCHING_THRESHOLD = 0.7;

  float track_thresh;
  float match_thresh;
  int frame_id;
  float det_thresh;
  int buffer_size;
  int max_time_lost;

  KalmanFilter kalman_filter;
  std::vector<std::shared_ptr<KalmanBBoxTrack>> tracked_tracks;
  std::vector<std::shared_ptr<KalmanBBoxTrack>> lost_tracks;
  std::vector<std::shared_ptr<KalmanBBoxTrack>> removed_tracks;

  LinearAssignment linear_assignment;

  std::vector<KalmanBBoxTrack> extract_kalman_bbox_tracks(
      const Eigen::MatrixXf dets, const Eigen::VectorXf scores_keep,
      const std::vector<types::FaceDetectRes> &detections);
  Eigen::MatrixXf select_matrix_rows_by_indices(const Eigen::MatrixXf matrix,
                                                const std::vector<int> indices);

  std::pair<std::vector<KalmanBBoxTrack>, std::vector<KalmanBBoxTrack>>
  filter_and_partition_detections(const Eigen::MatrixXf &output_results,
                                  const std::vector<types::FaceDetectRes> &detections);
  std::pair<std::vector<std::shared_ptr<KalmanBBoxTrack>>,
            std::vector<std::shared_ptr<KalmanBBoxTrack>>>

  partition_tracks_by_activation();
  std::tuple<std::vector<std::pair<int, int>>, std::set<int>, std::set<int>>

  assign_tracks_to_detections(
      const std::vector<std::shared_ptr<KalmanBBoxTrack>> tracks,
      const std::vector<KalmanBBoxTrack> detections, double thresh);

  void update_tracks_from_detections(
      std::vector<std::shared_ptr<KalmanBBoxTrack>> &tracks,
      const std::vector<KalmanBBoxTrack> detections,
      const std::vector<std::pair<int, int>> track_detection_pair_indices,
      std::vector<std::shared_ptr<KalmanBBoxTrack>> &reacquired_tracked_tracks,
      std::vector<std::shared_ptr<KalmanBBoxTrack>> &activated_tracks);

  std::vector<std::shared_ptr<KalmanBBoxTrack>> extract_active_tracks(
      const std::vector<std::shared_ptr<KalmanBBoxTrack>> &tracks,
      std::set<int> unpaired_track_indices);

  void flag_unpaired_tracks_as_lost(
      std::vector<std::shared_ptr<KalmanBBoxTrack>> &currently_tracked_tracks,
      std::vector<std::shared_ptr<KalmanBBoxTrack>> &lost_tracks,
      std::set<int> unpaired_track_indices);

  void prune_and_merge_tracked_tracks(
      std::vector<std::shared_ptr<KalmanBBoxTrack>> &reacquired_tracked_tracks,
      std::vector<std::shared_ptr<KalmanBBoxTrack>> &activated_tracks);

  void handle_lost_and_removed_tracks(
      std::vector<std::shared_ptr<KalmanBBoxTrack>> &removed_tracks,
      std::vector<std::shared_ptr<KalmanBBoxTrack>> &lost_tracks);

public:
  void TrackFace(std::vector<types::FaceDetectRes> &detections);
};
