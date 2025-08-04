#include "BYTETracker.h"
BYTETracker::BYTETracker(float track_thresh, int track_buffer,
                         float match_thresh, int frame_rate)
    : track_thresh(track_thresh), match_thresh(match_thresh), frame_id(0),
      det_thresh(track_thresh + MIN_KEEP_THRESH),
      buffer_size(
          static_cast<int>(frame_rate / BASE_FRAME_RATE * track_buffer)),
      max_time_lost(buffer_size), kalman_filter(KalmanFilter()) {
  tracked_tracks.clear();
  lost_tracks.clear();
  removed_tracks.clear();
  BaseTrack::reset_count();
}

std::vector<KalmanBBoxTrack> BYTETracker::extract_kalman_bbox_tracks(
    const Eigen::MatrixXf dets, const Eigen::VectorXf scores_keep,
    const std::vector<types::FaceDetectRes> &detections) {
  std::vector<KalmanBBoxTrack> result;
  if (dets.rows() > 0) {
    for (int i = 0; i < dets.rows(); ++i) {
      Eigen::Vector4f tlwh = dets.row(i);
      result.push_back(KalmanBBoxTrack(
          std::vector<float>{tlwh[0], tlwh[1], tlwh[2], tlwh[3]},
          scores_keep[i], detections[i]));
    }
  }

  return result;
}

Eigen::MatrixXf
BYTETracker::select_matrix_rows_by_indices(const Eigen::MatrixXf matrix,
                                           const std::vector<int> indices) {
  Eigen::MatrixXf result(indices.size(), matrix.cols());
  for (int i = 0; i < indices.size(); ++i) {
    result.row(i) = matrix.row(indices[i]);
  }
  return result;
}

std::pair<std::vector<KalmanBBoxTrack>, std::vector<KalmanBBoxTrack>>
BYTETracker::filter_and_partition_detections(
    const Eigen::MatrixXf &output_results,
    const std::vector<types::FaceDetectRes> &detections) {
  Eigen::VectorXf scores;
  Eigen::MatrixXf bboxes;
  scores = output_results.col(4);
  bboxes = output_results.leftCols(4);
  std::vector<int> indices_high_thresh, indices_low_thresh;
  for (int i = 0; i < scores.size(); ++i) {
    if (scores(i) > this->track_thresh) {
      indices_high_thresh.push_back(i);
    } else if (MIN_KEEP_THRESH < scores(i) && scores(i) < this->track_thresh) {
      indices_low_thresh.push_back(i);
    }
  }
  std::vector<KalmanBBoxTrack> detections_high = extract_kalman_bbox_tracks(
      select_matrix_rows_by_indices(bboxes, indices_high_thresh),
      select_matrix_rows_by_indices(scores, indices_high_thresh), detections);
  std::vector<KalmanBBoxTrack> detections_low = extract_kalman_bbox_tracks(
      select_matrix_rows_by_indices(bboxes, indices_low_thresh),
      select_matrix_rows_by_indices(scores, indices_low_thresh), detections);

  return {detections_high, detections_low};
}
std::pair<std::vector<std::shared_ptr<KalmanBBoxTrack>>,
          std::vector<std::shared_ptr<KalmanBBoxTrack>>>
BYTETracker::partition_tracks_by_activation() {
  std::vector<std::shared_ptr<KalmanBBoxTrack>> inactive_tracked_tracks;
  std::vector<std::shared_ptr<KalmanBBoxTrack>> active_tracked_tracks;
  for (auto &track : this->tracked_tracks) {
    if (track->get_is_activated()) {
      active_tracked_tracks.push_back(track);
    } else {
      inactive_tracked_tracks.push_back(track);
    }
  }

  return {inactive_tracked_tracks, active_tracked_tracks};
}
std::tuple<std::vector<std::pair<int, int>>, std::set<int>, std::set<int>>
BYTETracker::assign_tracks_to_detections(
    const std::vector<std::shared_ptr<KalmanBBoxTrack>> tracks,
    const std::vector<KalmanBBoxTrack> detections, double thresh) {
  std::vector<KalmanBBoxTrack> track_instances;
  track_instances.reserve(tracks.size());
  for (const auto &ptr : tracks) {
    track_instances.push_back(*ptr);
  }
  Eigen::MatrixXd distances = iou_distance(track_instances, detections);
  return this->linear_assignment.linear_assignment(distances, thresh);
}
void BYTETracker::update_tracks_from_detections(
    std::vector<std::shared_ptr<KalmanBBoxTrack>> &tracks,
    const std::vector<KalmanBBoxTrack> detections,
    const std::vector<std::pair<int, int>> track_detection_pair_indices,
    std::vector<std::shared_ptr<KalmanBBoxTrack>> &reacquired_tracked_tracks,
    std::vector<std::shared_ptr<KalmanBBoxTrack>> &activated_tracks) {

  for (const auto match : track_detection_pair_indices) {
    if (tracks[match.first]->get_state() == TrackState::Tracked) {
      tracks[match.first]->update(detections[match.second], this->frame_id,
                                  detections[match.second].detection);
      activated_tracks.push_back(tracks[match.first]);
    } else {
      tracks[match.first]->re_activate(detections[match.second], this->frame_id,
                                       detections[match.second].detection,
                                       false);
      reacquired_tracked_tracks.push_back(tracks[match.first]);
    }
  }
}

std::vector<std::shared_ptr<KalmanBBoxTrack>>
BYTETracker::extract_active_tracks(
    const std::vector<std::shared_ptr<KalmanBBoxTrack>> &tracks,
    std::set<int> unpaired_track_indices) {
  std::vector<std::shared_ptr<KalmanBBoxTrack>> currently_tracked_tracks;
  for (int i : unpaired_track_indices) {
    if (i < tracks.size() && tracks[i]->get_state() == TrackState::Tracked) {
      currently_tracked_tracks.push_back(tracks[i]);
    }
  }
  return currently_tracked_tracks;
}

void BYTETracker::flag_unpaired_tracks_as_lost(
    std::vector<std::shared_ptr<KalmanBBoxTrack>> &currently_tracked_tracks,
    std::vector<std::shared_ptr<KalmanBBoxTrack>> &lost_tracks,
    std::set<int> unpaired_track_indices) {
  for (int i : unpaired_track_indices) {
    if (i < currently_tracked_tracks.size() &&
        currently_tracked_tracks[i]->get_state() != TrackState::Lost) {
      currently_tracked_tracks[i]->mark_lost();
      lost_tracks.push_back(currently_tracked_tracks[i]);
    }
  }
}

void BYTETracker::prune_and_merge_tracked_tracks(
    std::vector<std::shared_ptr<KalmanBBoxTrack>> &reacquired_tracked_tracks,
    std::vector<std::shared_ptr<KalmanBBoxTrack>> &activated_tracks) {
  std::vector<std::shared_ptr<KalmanBBoxTrack>> filtered_tracked_tracks;
  for (std::shared_ptr<KalmanBBoxTrack> track : this->tracked_tracks) {
    if (track->get_state() == TrackState::Tracked) {
      filtered_tracked_tracks.push_back(track);
    }
  }
  this->tracked_tracks = filtered_tracked_tracks;
  this->tracked_tracks = join_tracks(this->tracked_tracks, activated_tracks);
  this->tracked_tracks =
      join_tracks(this->tracked_tracks, reacquired_tracked_tracks);
}

void BYTETracker::handle_lost_and_removed_tracks(
    std::vector<std::shared_ptr<KalmanBBoxTrack>> &removed_tracks,
    std::vector<std::shared_ptr<KalmanBBoxTrack>> &lost_tracks) {
  for (std::shared_ptr<KalmanBBoxTrack> track : this->lost_tracks) {
    if (this->frame_id - track->end_frame() > this->max_time_lost) {
      track->mark_removed();
      removed_tracks.push_back(track);
    }
  }
  this->lost_tracks = sub_tracks(this->lost_tracks, this->tracked_tracks);
  this->lost_tracks.insert(this->lost_tracks.end(), lost_tracks.begin(),
                           lost_tracks.end());
  this->lost_tracks = sub_tracks(this->lost_tracks, this->removed_tracks);
  this->removed_tracks.clear();
}

void BYTETracker::TrackFace(std::vector<types::FaceDetectRes> &detections) {

  Eigen::MatrixXf output_results(detections.size(), 5);
  for (size_t i = 0; i < detections.size(); i++) {
    output_results(i, 0) = detections[i].x1;
    output_results(i, 1) = detections[i].y1;
    output_results(i, 2) = detections[i].x2 - detections[i].x1;
    output_results(i, 3) = detections[i].y2 - detections[i].y1;
    output_results(i, 4) = detections[i].score;
  }

  this->frame_id += 1;
  std::vector<std::shared_ptr<KalmanBBoxTrack>> reacquired_tracked_tracks,
      activated_tracks, lost_tracks, removed_tracks;

  auto [high_confidence_detections, lower_confidence_detections] =
      filter_and_partition_detections(output_results, detections);

  auto [inactive_tracked_tracks, active_tracked_tracks] =
      partition_tracks_by_activation();

  std::vector<std::shared_ptr<KalmanBBoxTrack>> track_pool =
      join_tracks(active_tracked_tracks, this->lost_tracks);

  KalmanBBoxTrack::multi_predict(track_pool);

  auto [track_detection_pair_indices, unpaired_track_indices,
        unpaired_detection_indices] =
      assign_tracks_to_detections(track_pool, high_confidence_detections,
                                  this->match_thresh);
  update_tracks_from_detections(track_pool, high_confidence_detections,
                                track_detection_pair_indices,
                                reacquired_tracked_tracks, activated_tracks);
  auto currently_tracked_tracks =
      extract_active_tracks(track_pool, unpaired_track_indices);
  std::tie(track_detection_pair_indices, unpaired_track_indices, std::ignore) =
      assign_tracks_to_detections(currently_tracked_tracks,
                                  lower_confidence_detections,
                                  LOWER_CONFIDENCE_MATCHING_THRESHOLD);
  update_tracks_from_detections(currently_tracked_tracks,
                                lower_confidence_detections,
                                track_detection_pair_indices,
                                reacquired_tracked_tracks, activated_tracks);
  flag_unpaired_tracks_as_lost(currently_tracked_tracks, lost_tracks,
                               unpaired_track_indices);
  std::vector<KalmanBBoxTrack> filtered_detections;
  for (int i : unpaired_detection_indices) {
    filtered_detections.push_back(high_confidence_detections[i]);
  }
  high_confidence_detections = filtered_detections;
  std::tie(track_detection_pair_indices, unpaired_track_indices,
           unpaired_detection_indices) =
      assign_tracks_to_detections(inactive_tracked_tracks,
                                  high_confidence_detections,
                                  ACTIVATION_MATCHING_THRESHOLD);
  for (auto [track_idx, det_idx] : track_detection_pair_indices) {
    inactive_tracked_tracks[track_idx]->update(
        high_confidence_detections[det_idx], this->frame_id,
        high_confidence_detections[det_idx].detection);
    activated_tracks.push_back(inactive_tracked_tracks[track_idx]);
  }
  for (int i : unpaired_track_indices) {
    inactive_tracked_tracks[i]->mark_removed();
    removed_tracks.push_back(inactive_tracked_tracks[i]);
  }
  for (int i : unpaired_detection_indices) {
    if (high_confidence_detections[i].get_score() >= this->det_thresh) {
      high_confidence_detections[i].activate(this->kalman_filter,
                                             this->frame_id);
      activated_tracks.push_back(
          std::make_shared<KalmanBBoxTrack>(high_confidence_detections[i]));
    }
  }
  prune_and_merge_tracked_tracks(reacquired_tracked_tracks, activated_tracks);
  handle_lost_and_removed_tracks(removed_tracks, lost_tracks);
  this->removed_tracks.insert(this->removed_tracks.end(),
                              removed_tracks.begin(), removed_tracks.end());
  std::tie(this->tracked_tracks, this->lost_tracks) =
      remove_duplicate_tracks(this->tracked_tracks, this->lost_tracks);
  detections.clear();
  for (std::shared_ptr<KalmanBBoxTrack> track : this->tracked_tracks) {
    if (track->get_is_activated()) {
      track->detection.id_tracking = track->get_track_id();
      detections.push_back(track->detection);
    }
  }
}
