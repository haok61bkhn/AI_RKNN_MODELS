#pragma once

#include <Eigen/Dense>
#include <cfloat>
#include <iostream>

class HungarianAlgorithmEigen {
public:
  HungarianAlgorithmEigen();
  ~HungarianAlgorithmEigen();

  double solve_assignment_problem(Eigen::MatrixXd &dist_matrix,
                                  Eigen::VectorXi &assignment);

private:
  const int NOT_FOUND_VALUE = -1;
  const int DEFAULT_ASSIGNMENT_VALUE = -1;

  Eigen::MatrixXd dist_matrix;

  Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic> star_matrix,
      new_star_matrix, prime_matrix;
  Eigen::Array<bool, Eigen::Dynamic, 1> covered_columns, covered_rows;

  Eigen::Index find_star_in_column(int col);

  Eigen::Index find_prime_in_row(int row);

  void update_star_and_prime_matrices(int row, int col);

  void reduce_matrix_by_minimum_value();

  void cover_columns_lacking_stars();

  void execute_hungarian_algorithm();

  void init_helper_arrays(int num_rows, int num_columns);

  void construct_assignment_vector(Eigen::VectorXi &assignment);

  double calculate_total_cost(Eigen::VectorXi &assignment);
};
