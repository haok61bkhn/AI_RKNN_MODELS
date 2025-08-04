#pragma once

#include "BaseTrack.h"
#include "KalmanFilter.h"
#include "types.h"
#include <Eigen/Dense>
#include <iostream>
#include <memory>
#include <vector>

class KalmanBBoxTrack : public BaseTrack {
private:
  static KalmanFilter shared_kalman;

public:
  Eigen::Vector4d _tlwh;
  KalmanFilter kalman_filter;
  Eigen::VectorXd mean;
  Eigen::MatrixXd covariance;
  types::FaceDetectRes detection;
  int tracklet_len;

  KalmanBBoxTrack();

  KalmanBBoxTrack(const std::vector<float> tlwh, float score,
                  const types::FaceDetectRes &detection);

  KalmanBBoxTrack(const types::FaceDetectRes &detection);

  static void
  multi_predict(std::vector<std::shared_ptr<KalmanBBoxTrack>> &tracks);

  Eigen::VectorXd tlwh_to_xyah(Eigen::VectorXd tlwh);

  void activate(KalmanFilter &kalman_filter, int frame_id);

  void update_track(const KalmanBBoxTrack &new_track, int frame_id,
                    const types::FaceDetectRes &detection,
                    bool new_id = false);

  void re_activate(const KalmanBBoxTrack &new_track, int frame_id,
                   const types::FaceDetectRes &detection,
                   bool new_id = false);

  void update(const KalmanBBoxTrack &new_track, int frame_id,
              const types::FaceDetectRes &detection);

  static Eigen::Vector4d tlwh_to_tlbr(const Eigen::Vector4d tlwh);

  static Eigen::Vector4d tlbr_to_tlwh(const Eigen::Vector4d tlbr);

  Eigen::Vector4d tlwh() const;

  Eigen::Vector4d tlbr() const;
};
