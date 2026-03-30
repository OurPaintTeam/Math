//
// Created by Eugene Bychkov on 18.05.2025.
//

#ifndef TASKMATRIX_H
#define TASKMATRIX_H

#include "Matrix.h"
#include "Task.h"
class TaskMatrix : public Task {
public:
  virtual Matrix<> gradient() const = 0;
  virtual Matrix<> hessian() const = 0;
};
#endif //TASKMATRIX_H
