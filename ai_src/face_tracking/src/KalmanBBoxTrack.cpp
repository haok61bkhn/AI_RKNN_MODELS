#include "KalmanBBoxTrack.h"

KalmanFilter KalmanBBoxTrack::shared_kalman;

KalmanBBoxTrack::KalmanBBoxTrack()
    : BaseTrack(), _tlwh(Eigen::Vector4d::Zero()),
      kalman_filter(KalmanFilter()), mean(Eigen::VectorXd()),
      covariance(Eigen::MatrixXd()), tracklet_len(0) {}

KalmanBBoxTrack::KalmanBBoxTrack(const std::vector<float> tlwh, float score,
                                 const types::FaceDetectRes &detection)
    : BaseTrack(score),
      _tlwh(tlwh.size() == 4
                ? Eigen::Vector4d(tlwh[0], tlwh[1], tlwh[2], tlwh[3])
                : Eigen::Vector4d::Zero()),
      kalman_filter(KalmanFilter()), mean(Eigen::VectorXd()),
      covariance(Eigen::MatrixXd()), tracklet_len(0) {
  if (tlwh.size() != 4) {
    throw std::invalid_argument("tlwh vector must contain exactly 4 values.");
  }
  this->detection = detection;
}

KalmanBBoxTrack::KalmanBBoxTrack(const types::FaceDetectRes& detection)
    : BaseTrack(detection.score),
      _tlwh(Eigen::Vector4d(detection.x1, detection.y1, 
                           detection.x2 - detection.x1, 
                           detection.y2 - detection.y1)),
      kalman_filter(KalmanFilter()), mean(Eigen::VectorXd()),
      covariance(Eigen::MatrixXd()), tracklet_len(0), detection(detection) {}

void KalmanBBoxTrack::multi_predict(
    std::vector<std::shared_ptr<KalmanBBoxTrack>> &tracks) {
  if (tracks.empty()) {
    return;
  }

  std::vector<Eigen::VectorXd> multi_means;
  std::vector<Eigen::MatrixXd> multi_covariances;

  for (const auto track : tracks) {
    multi_means.push_back(track->mean);
    multi_covariances.push_back(track->covariance);
  }

  for (size_t i = 0; i < tracks.size(); ++i) {
    if (tracks[i]->get_state() != TrackState::Tracked) {
      multi_means[i](7) = 0;
    }
  }

  Eigen::MatrixXd means_matrix(multi_means.size(), multi_means[0].size());
  for (size_t i = 0; i < multi_means.size(); ++i) {
    means_matrix.row(i) = multi_means[i];
  }
  int n = (int)multi_covariances[0].rows();
  Eigen::MatrixXd covariances_matrix(n, n * multi_covariances.size());
  for (size_t i = 0; i < multi_covariances.size(); ++i) {
    covariances_matrix.middleCols(i * n, n) = multi_covariances[i];
  }
  Eigen::MatrixXd predicted_means, predicted_covariances;
  std::tie(predicted_means, predicted_covariances) =
      shared_kalman.multi_predict(means_matrix, covariances_matrix);
  for (size_t i = 0; i < tracks.size(); ++i) {
    tracks[i]->mean = predicted_means.col(i);
  }
  int pcr = (int)predicted_covariances.rows();
  size_t num_matrices = predicted_covariances.cols() / pcr;
  for (size_t i = 0; i < num_matrices; ++i) {
    Eigen::MatrixXd block = predicted_covariances.middleCols(i * pcr, pcr);
    tracks[i]->covariance = block;
  }
}

Eigen::VectorXd KalmanBBoxTrack::tlwh_to_xyah(Eigen::VectorXd tlwh) {
  Eigen::VectorXd ret = tlwh;
  ret.head(2) += ret.segment(2, 2) / 2.0;
  ret(2) /= ret(3);
  return ret;
}

void KalmanBBoxTrack::activate(KalmanFilter &kalman_filter, int frame_id) {
  this->kalman_filter = kalman_filter;
  this->track_id = BaseTrack::next_id();
  std::tie(this->mean, this->covariance) =
      this->kalman_filter.initiate(this->tlwh_to_xyah(this->_tlwh));
  this->tracklet_len = 0;
  this->state = TrackState::Tracked;
  this->is_activated = (frame_id == 1);
  this->frame_id = frame_id;
  this->start_frame = frame_id;
}

void KalmanBBoxTrack::update_track(const KalmanBBoxTrack &new_track,
                                   int frame_id,
                                   const types::FaceDetectRes &detection,
                                   bool new_id) {
  this->frame_id = frame_id;
  this->tracklet_len++;
  std::tie(this->mean, this->covariance) = this->kalman_filter.update(
      this->mean, this->covariance, this->tlwh_to_xyah(new_track.tlwh()));
  this->state = TrackState::Tracked;
  this->is_activated = true;
  if (new_id) {
    this->track_id = BaseTrack::next_id();
  }
  this->score = new_track.get_score();
  this->detection = detection;
}

void KalmanBBoxTrack::re_activate(const KalmanBBoxTrack &new_track,
                                  int frame_id, 
                                  const types::FaceDetectRes &detection,
                                  bool new_id) {
  update_track(new_track, frame_id, detection, new_id);
}

void KalmanBBoxTrack::update(const KalmanBBoxTrack &new_track, int frame_id,
                             const types::FaceDetectRes &detection) {
  update_track(new_track, frame_id, detection);
}

Eigen::Vector4d KalmanBBoxTrack::tlwh_to_tlbr(const Eigen::Vector4d tlwh) {
  Eigen::Vector4d ret = tlwh;
  ret.tail<2>() += ret.head<2>();
  return ret;
}

Eigen::Vector4d KalmanBBoxTrack::tlbr_to_tlwh(const Eigen::Vector4d tlbr) {
  Eigen::Vector4d ret = tlbr;
  ret.tail<2>() -= ret.head<2>();
  return ret;
}

Eigen::Vector4d KalmanBBoxTrack::tlwh() const {
  if (mean.isZero(0)) {
    return _tlwh;
  }

  Eigen::Vector4d ret = mean.head(4);
  ret[2] *= ret[3];
  ret[0] -= ret[2] / 2.0;
  ret[1] -= ret[3] / 2.0;

  return ret;
}

Eigen::Vector4d KalmanBBoxTrack::tlbr() const {
  return this->tlwh_to_tlbr(this->tlwh());
}