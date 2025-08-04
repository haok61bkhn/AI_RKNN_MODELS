#include "HungarianAlgorithmEigen.h"

HungarianAlgorithmEigen::HungarianAlgorithmEigen() {}

HungarianAlgorithmEigen::~HungarianAlgorithmEigen() {}

Eigen::Index HungarianAlgorithmEigen::find_star_in_column(int col) {
  for (Eigen::Index i = 0; i < star_matrix.rows(); ++i) {
    if (star_matrix(i, col)) {
      return i;
    }
  }
  return NOT_FOUND_VALUE;
}

Eigen::Index HungarianAlgorithmEigen::find_prime_in_row(int row) {
  for (Eigen::Index j = 0; j < prime_matrix.cols(); ++j) {
    if (prime_matrix(row, j)) {
      return j;
    }
  }
  return NOT_FOUND_VALUE;
}

void HungarianAlgorithmEigen::update_star_and_prime_matrices(int row, int col) {
  new_star_matrix = star_matrix;
  new_star_matrix(row, col) = true;
  while (true) {
    Eigen::Index star_row, prime_col;
    if (!star_matrix.col(col).any()) {
      break;
    }
    star_row = find_star_in_column(col);
    new_star_matrix(star_row, col) = false;
    if (!prime_matrix.row(star_row).any()) {
      break;
    }
    prime_col = find_prime_in_row(star_row);
    new_star_matrix(star_row, prime_col) = true;
    col = prime_col;
  }
  star_matrix = new_star_matrix;
  prime_matrix.setConstant(false);
  covered_rows.setConstant(false);
}

void HungarianAlgorithmEigen::reduce_matrix_by_minimum_value() {
  int num_rows = covered_rows.size();
  int num_columns = covered_columns.size();
  Eigen::ArrayXXd masked_array =
      dist_matrix.array() +
      DBL_MAX *
          (covered_rows.replicate(1, num_columns).cast<double>() +
           covered_columns.transpose().replicate(num_rows, 1).cast<double>());
  double min_uncovered_value = masked_array.minCoeff();
  Eigen::ArrayXXd row_adjustments =
      covered_rows.cast<double>() * min_uncovered_value;
  Eigen::ArrayXXd col_adjustments =
      (1.0 - covered_columns.cast<double>()) * min_uncovered_value;
  dist_matrix += (row_adjustments.replicate(1, num_columns) -
                  col_adjustments.transpose().replicate(num_rows, 1))
                     .matrix();
}

void HungarianAlgorithmEigen::cover_columns_lacking_stars() {
  int num_rows = dist_matrix.rows();
  int num_columns = dist_matrix.cols();
  bool zeros_found = true;
  while (zeros_found) {
    zeros_found = false;
    for (int col = 0; col < num_columns; col++) {
      if (covered_columns(col))
        continue;
      Eigen::Array<bool, Eigen::Dynamic, 1> uncovered_zeros_in_column =
          (dist_matrix.col(col).array().abs() < DBL_EPSILON) && !covered_rows;
      Eigen::Index row;
      double max_in_uncovered_zeros =
          uncovered_zeros_in_column.cast<double>().maxCoeff(&row);
      if (max_in_uncovered_zeros == 1.0) {
        prime_matrix(row, col) = true;
        Eigen::Index star_col;
        bool has_star = star_matrix.row(row).maxCoeff(&star_col);
        if (!has_star) {
          update_star_and_prime_matrices(row, col);
          covered_columns = (star_matrix.colwise().any()).transpose();
          return;
        } else {
          covered_rows(row) = true;
          covered_columns(star_col) = false;
          zeros_found = true;
          break;
        }
      }
    }
  }
  reduce_matrix_by_minimum_value();
  cover_columns_lacking_stars();
}

void HungarianAlgorithmEigen::execute_hungarian_algorithm() {
  if (dist_matrix.rows() <= dist_matrix.cols()) {
    for (int row = 0; row < dist_matrix.rows(); ++row) {
      double min_value = dist_matrix.row(row).minCoeff();
      dist_matrix.row(row).array() -= min_value;
      Eigen::ArrayXd current_row = dist_matrix.row(row).array();
      Eigen::Array<bool, Eigen::Dynamic, 1> uncovered_zeros =
          (current_row.abs() < DBL_EPSILON) && !covered_columns;
      Eigen::Index col;
      double max_in_uncovered_zeros =
          uncovered_zeros.cast<double>().maxCoeff(&col);
      if (max_in_uncovered_zeros == 1.0) {
        star_matrix(row, col) = true;
        covered_columns(col) = true;
      }
    }
  } else {
    for (int col = 0; col < dist_matrix.cols(); ++col) {
      double min_value = dist_matrix.col(col).minCoeff();
      dist_matrix.col(col).array() -= min_value;
      Eigen::ArrayXd current_column = dist_matrix.col(col).array();
      Eigen::Array<bool, Eigen::Dynamic, 1> uncovered_zeros =
          (current_column.abs() < DBL_EPSILON) && !covered_rows;
      Eigen::Index row;
      double max_in_uncovered_zeros =
          uncovered_zeros.cast<double>().maxCoeff(&row);
      if (max_in_uncovered_zeros == 1.0) {
        star_matrix(row, col) = true;
        covered_columns(col) = true;
        covered_rows(row) = true;
      }
    }
    for (int row = 0; row < dist_matrix.rows(); ++row) {
      covered_rows(row) = false;
    }
  }
  if (covered_columns.count() !=
      std::min(dist_matrix.rows(), dist_matrix.cols())) {
    cover_columns_lacking_stars();
  }
}

void HungarianAlgorithmEigen::init_helper_arrays(int num_rows,
                                                 int num_columns) {
  covered_columns =
      Eigen::Array<bool, Eigen::Dynamic, 1>::Constant(num_columns, false);
  covered_rows =
      Eigen::Array<bool, Eigen::Dynamic, 1>::Constant(num_rows, false);
  star_matrix = Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic>::Constant(
      num_rows, num_columns, false);
  new_star_matrix =
      Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic>::Constant(
          num_rows, num_columns, false);
  prime_matrix = Eigen::Array<bool, Eigen::Dynamic, Eigen::Dynamic>::Constant(
      num_rows, num_columns, false);
}

void HungarianAlgorithmEigen::construct_assignment_vector(
    Eigen::VectorXi &assignment) {
  for (int row = 0; row < star_matrix.rows(); ++row) {
    Eigen::Index col;
    bool has_star = star_matrix.row(row).maxCoeff(&col);
    if (has_star) {
      assignment[row] = col;
    } else {
      assignment[row] = DEFAULT_ASSIGNMENT_VALUE;
    }
  }
}

double
HungarianAlgorithmEigen::calculate_total_cost(Eigen::VectorXi &assignment) {
  double total_cost = 0.0;
  for (int row = 0; row < dist_matrix.rows(); ++row) {
    if (assignment(row) >= 0) {
      total_cost += dist_matrix(row, assignment(row));
    }
  }
  return total_cost;
}

double
HungarianAlgorithmEigen::solve_assignment_problem(Eigen::MatrixXd &dist_matrix,
                                                  Eigen::VectorXi &assignment) {
  if (dist_matrix.array().minCoeff() < 0) {
    std::cerr << "All matrix elements have to be non-negative." << std::endl;
  }
  this->dist_matrix = dist_matrix;
  init_helper_arrays(dist_matrix.rows(), dist_matrix.cols());
  execute_hungarian_algorithm();
  assignment.setConstant(DEFAULT_ASSIGNMENT_VALUE);
  construct_assignment_vector(assignment);
  return calculate_total_cost(assignment);
}