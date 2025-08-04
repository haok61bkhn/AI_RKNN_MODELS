#include "KalmanFilter.h"

KalmanFilter::KalmanFilter() {
    ndim = 4;  // State space dimension (x, y, width, height)
    double dt = 1.0;  // Time step, assuming time in seconds
    motion_mat = Eigen::MatrixXd::Identity(2 * ndim, 2 * ndim);
    motion_mat.block(0, ndim, ndim, ndim) = dt * Eigen::MatrixXd::Identity(ndim, ndim);
    update_mat = Eigen::MatrixXd::Identity(ndim, 2 * ndim);
    std_weight_position = 1. / 20.;
    std_weight_velocity = 1. / 160.;
}

KalmanFilter::~KalmanFilter() {}

Eigen::VectorXd KalmanFilter::create_std(const Eigen::VectorXd& mean) {
    Eigen::VectorXd std_devs(8);
    std_devs(0) = std_weight_position * mean(3);
    std_devs(1) = std_weight_position * mean(3);
    std_devs(2) = 1e-2;
    std_devs(3) = std_weight_position * mean(3);
    std_devs(4) = std_weight_velocity * mean(3);
    std_devs(5) = std_weight_velocity * mean(3);
    std_devs(6) = 1e-5;  // Small fixed standard deviation for velocity in width
    std_devs(7) = std_weight_velocity * mean(3);

    return std_devs;
}

std::pair<Eigen::VectorXd, Eigen::MatrixXd> KalmanFilter::initiate(const Eigen::VectorXd& measurement) {
    Eigen::VectorXd mean = Eigen::VectorXd::Zero(2 * ndim);
    mean.head(ndim) = measurement;
    Eigen::VectorXd std_devs = create_std(mean);
    Eigen::MatrixXd covariance = Eigen::MatrixXd::Zero(2 * ndim, 2 * ndim);
    for (int i = 0; i < 2 * ndim; i++) {
        covariance(i, i) = std_devs(i) * std_devs(i);
    }

    return { mean, covariance };
}

std::pair<Eigen::VectorXd, Eigen::MatrixXd> KalmanFilter::project(const Eigen::VectorXd& mean, const Eigen::MatrixXd& covariance) {
    Eigen::VectorXd projected_std = create_std(mean).head(ndim);
    Eigen::MatrixXd innovation_cov = Eigen::MatrixXd::Zero(ndim, ndim);
    for (int i = 0; i < ndim; i++) {
        innovation_cov(i, i) = projected_std(i) * projected_std(i);
    }
    Eigen::VectorXd new_mean = this->update_mat * mean;
    Eigen::MatrixXd new_cov = this->update_mat * covariance * this->update_mat.transpose() + innovation_cov;
    return { new_mean, new_cov };
}
std::pair<Eigen::MatrixXd, Eigen::MatrixXd> KalmanFilter::multi_predict(const Eigen::MatrixXd& means, const Eigen::MatrixXd& covariances) {
    int n_tracks = means.rows();
    Eigen::MatrixXd new_means(2 * ndim, n_tracks);
    Eigen::MatrixXd new_covs(2 * ndim, 2 * ndim * n_tracks);
    for (int i = 0; i < n_tracks; ++i) {
        Eigen::VectorXd motion_std = create_std(means.row(i));
        Eigen::MatrixXd motion_cov = Eigen::MatrixXd::Zero(2 * ndim, 2 * ndim);
        for (int j = 0; j < 2 * ndim; j++) {
            motion_cov(j, j) = motion_std(j) * motion_std(j);
        }
        new_means.col(i) = motion_mat * means.row(i).transpose();
        new_covs.block(0, 2 * ndim * i, 2 * ndim, 2 * ndim) = motion_mat * covariances.block(0, 2 * ndim * i, 2 * ndim, 2 * ndim) * motion_mat.transpose() + motion_cov;
    }

    return { new_means, new_covs };
}

std::pair<Eigen::VectorXd, Eigen::MatrixXd> KalmanFilter::update(
    const Eigen::VectorXd& mean,
    const Eigen::MatrixXd& covariance,
    const Eigen::VectorXd& measurement)
{
    Eigen::VectorXd projected_mean;
    Eigen::MatrixXd projected_cov;
    std::tie(projected_mean, projected_cov) = project(mean, covariance);
    Eigen::LLT<Eigen::MatrixXd> cho_factor(projected_cov);
    if (cho_factor.info() != Eigen::Success) {
        throw std::runtime_error("Decomposition failed!");
    }
    Eigen::MatrixXd kalman_gain = cho_factor.solve(this->update_mat * covariance).transpose();
    Eigen::VectorXd new_mean = mean + kalman_gain * (measurement - projected_mean);
    Eigen::MatrixXd new_covariance = covariance - kalman_gain * projected_cov * kalman_gain.transpose();
    return { new_mean, new_covariance };
}
