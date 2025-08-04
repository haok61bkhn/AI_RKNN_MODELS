#include "LinearAssignment.h"

std::tuple<std::vector<std::pair<int, int>>, std::set<int>, std::set<int>> LinearAssignment::indices_to_matches(
    const Eigen::MatrixXd& cost_matrix, const Eigen::MatrixXi& indices, double thresh)
{
    if (cost_matrix.rows() <= 0 || cost_matrix.cols() <= 0) {
        throw std::invalid_argument("Cost matrix dimensions must be positive.");
    }
    std::vector<std::pair<int, int>> matches;
    std::set<int> unmatched_a, unmatched_b;
    int num_rows = cost_matrix.rows();
    int num_cols = cost_matrix.cols();
    for (int i = 0; i < num_rows; i++)
        unmatched_a.insert(i);
    for (int j = 0; j < num_cols; j++)
        unmatched_b.insert(j);
    for (int k = 0; k < indices.rows(); k++)
    {
        int i = indices(k, 0);
        int j = indices(k, 1);
        if (i != -1 && j != -1) {
            if (cost_matrix(i, j) <= thresh)
            {
                matches.push_back({ i, j });
                unmatched_a.erase(i);
                unmatched_b.erase(j);
            }
        }
    }
    return { matches, unmatched_a, unmatched_b };
}

std::tuple<std::vector<std::pair<int, int>>, std::set<int>, std::set<int>> LinearAssignment::linear_assignment(
    const Eigen::MatrixXd& cost_matrix, double thresh)
{
    int num_rows = cost_matrix.rows();
    int num_cols = cost_matrix.cols();
    if (num_rows == 0 || num_cols == 0)
    {
        std::set<int> unmatched_indices_first;
        std::set<int> unmatched_indices_second;
        for (int i = 0; i < num_rows; i++) {
            unmatched_indices_first.insert(i);
        }
        for (int i = 0; i < num_cols; i++) {
            unmatched_indices_second.insert(i);
        }
        return { {}, unmatched_indices_first, unmatched_indices_second };
    }
    Eigen::MatrixXd modified_cost_matrix = cost_matrix.unaryExpr([thresh](double val) {
        return (val > thresh) ? thresh + 1e-4 : val;
        });
    Eigen::VectorXi assignment = Eigen::VectorXi::Constant(modified_cost_matrix.rows(), -1);
    this->hungarian.solve_assignment_problem(modified_cost_matrix, assignment);
    Eigen::MatrixXi indices = Eigen::MatrixXi::Zero(num_rows, 2);
    indices.col(0) = Eigen::VectorXi::LinSpaced(num_rows, 0, num_rows - 1);
    indices.col(1) = assignment;
    return indices_to_matches(cost_matrix, indices, thresh);
}
