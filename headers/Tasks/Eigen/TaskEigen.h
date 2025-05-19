//
// Created by Eugene Bychkov on 18.05.2025.
//

#ifndef TASKEIGEN_H
#define TASKEIGEN_H
#include "Task.h"
#include <Eigen/Dense>
class TaskEigen: public Task {
  virtual Eigen::VectorXd gradient() const = 0;
  virtual Eigen::MatrixXd hessian() const = 0;
  virtual Eigen::MatrixXd jacobian() const = 0;
  virtual std::pair<Eigen::VectorXd, Eigen::MatrixXd> linearizeFunction() const = 0;
};
#endif //TASKEIGEN_H
