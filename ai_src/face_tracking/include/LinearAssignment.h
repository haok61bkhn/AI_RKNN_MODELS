#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <iostream>
#include <set>
#include <tuple>
#include <vector>

#include "HungarianAlgorithmEigen.h"

class LinearAssignment {
private:
  HungarianAlgorithmEigen hungarian;

public:
  std::tuple<std::vector<std::pair<int, int>>, std::set<int>, std::set<int>>
  indices_to_matches(const Eigen::MatrixXd &cost_matrix,
                     const Eigen::MatrixXi &indices, double thresh);

  std::tuple<std::vector<std::pair<int, int>>, std::set<int>, std::set<int>>
  linear_assignment(const Eigen::MatrixXd &cost_matrix, double thresh);
};
