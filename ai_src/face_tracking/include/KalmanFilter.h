#pragma once
#include "Export.h"
#include <Eigen/Cholesky>
#include <Eigen/Dense>
#include <iostream>
#include <stdexcept>

class BYTE_TRACK_EIGEN_API KalmanFilter {
public:
  KalmanFilter();

  ~KalmanFilter();

  Eigen::VectorXd create_std(const Eigen::VectorXd &mean);

  std::pair<Eigen::VectorXd, Eigen::MatrixXd>
  initiate(const Eigen::VectorXd &measurement);

  std::pair<Eigen::VectorXd, Eigen::MatrixXd>
  project(const Eigen::VectorXd &mean, const Eigen::MatrixXd &covariance);

  std::pair<Eigen::MatrixXd, Eigen::MatrixXd>
  multi_predict(const Eigen::MatrixXd &mean, const Eigen::MatrixXd &covariance);

  std::pair<Eigen::VectorXd, Eigen::MatrixXd>
  update(const Eigen::VectorXd &mean, const Eigen::MatrixXd &covariance,
         const Eigen::VectorXd &measurement);

private:
  int ndim;
  Eigen::MatrixXd motion_mat;
  Eigen::MatrixXd update_mat;
  double std_weight_position;
  double std_weight_velocity;
};
