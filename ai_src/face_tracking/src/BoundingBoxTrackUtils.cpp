#include "BoundingBoxTrackUtils.h"
std::vector<std::shared_ptr<KalmanBBoxTrack>>
join_tracks(const std::vector<std::shared_ptr<KalmanBBoxTrack>> track_list_a,
            const std::vector<std::shared_ptr<KalmanBBoxTrack>> track_list_b) {

  std::map<int, std::shared_ptr<KalmanBBoxTrack>> unique_tracks;
  for (auto &track : track_list_a) {
    unique_tracks[track->get_track_id()] = track;
  }
  for (auto &track : track_list_b) {
    unique_tracks[track->get_track_id()] = track;
  }
  std::vector<std::shared_ptr<KalmanBBoxTrack>> result;
  for (auto [key, value] : unique_tracks) {
    result.push_back(value);
  }

  return result;
}

std::vector<std::shared_ptr<KalmanBBoxTrack>>
sub_tracks(const std::vector<std::shared_ptr<KalmanBBoxTrack>> &track_list_a,
           const std::vector<std::shared_ptr<KalmanBBoxTrack>> &track_list_b) {
  std::unordered_set<int> track_ids_b;
  for (const auto &track : track_list_b) {
    track_ids_b.insert(track->get_track_id());
  }
  std::vector<std::shared_ptr<KalmanBBoxTrack>> result;
  for (const auto &track : track_list_a) {
    if (track_ids_b.find(track->get_track_id()) == track_ids_b.end()) {
      result.push_back(track);
    }
  }

  return result;
}

std::pair<std::vector<std::shared_ptr<KalmanBBoxTrack>>,
          std::vector<std::shared_ptr<KalmanBBoxTrack>>>
remove_duplicate_tracks(
    const std::vector<std::shared_ptr<KalmanBBoxTrack>> track_list_a,
    const std::vector<std::shared_ptr<KalmanBBoxTrack>> track_list_b) {
  std::vector<KalmanBBoxTrack> s_tracks_a_instances;
  s_tracks_a_instances.reserve(track_list_a.size());
  for (const auto &ptr : track_list_a) {
    s_tracks_a_instances.push_back(*ptr);
  }

  std::vector<KalmanBBoxTrack> s_tracks_b_instances;
  s_tracks_b_instances.reserve(track_list_b.size());
  for (const auto &ptr : track_list_b) {
    s_tracks_b_instances.push_back(*ptr);
  }
  Eigen::MatrixXd pairwise_distance =
      iou_distance(s_tracks_a_instances, s_tracks_b_instances);
  std::unordered_set<int> duplicates_a, duplicates_b;
  for (int i = 0; i < pairwise_distance.rows(); ++i) {
    for (int j = 0; j < pairwise_distance.cols(); ++j) {
      if (pairwise_distance(i, j) < 0.15) {
        int time_a = track_list_a[i]->get_frame_id() -
                     track_list_a[i]->get_start_frame();
        int time_b = track_list_b[j]->get_frame_id() -
                     track_list_b[j]->get_start_frame();
        if (time_a > time_b) {
          duplicates_b.insert(j);
        } else {
          duplicates_a.insert(i);
        }
      }
    }
  }
  std::vector<std::shared_ptr<KalmanBBoxTrack>> result_a, result_b;
  for (int i = 0; i < track_list_a.size(); ++i) {
    if (duplicates_a.find(i) == duplicates_a.end()) {
      result_a.push_back(track_list_a[i]);
    }
  }
  for (int j = 0; j < track_list_b.size(); ++j) {
    if (duplicates_b.find(j) == duplicates_b.end()) {
      result_b.push_back(track_list_b[j]);
    }
  }

  return {result_a, result_b};
}
