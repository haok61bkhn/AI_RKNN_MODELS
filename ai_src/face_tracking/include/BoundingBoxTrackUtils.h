#pragma once

#include "BoundingBoxIoUMatching.h"
#include "KalmanBBoxTrack.h"
#include <Eigen/Dense>
#include <map>
#include <unordered_set>
#include <vector>

std::vector<std::shared_ptr<KalmanBBoxTrack>>
join_tracks(const std::vector<std::shared_ptr<KalmanBBoxTrack>> track_list_a,
            const std::vector<std::shared_ptr<KalmanBBoxTrack>> track_list_b);

std::vector<std::shared_ptr<KalmanBBoxTrack>>
sub_tracks(const std::vector<std::shared_ptr<KalmanBBoxTrack>> &track_list_a,
           const std::vector<std::shared_ptr<KalmanBBoxTrack>> &track_list_b);

std::pair<std::vector<std::shared_ptr<KalmanBBoxTrack>>,
          std::vector<std::shared_ptr<KalmanBBoxTrack>>>
remove_duplicate_tracks(
    const std::vector<std::shared_ptr<KalmanBBoxTrack>> track_list_a,
    const std::vector<std::shared_ptr<KalmanBBoxTrack>> track_list_b);
