//
// Created by Eugene Bychkov on 18.05.2025.
//

#ifndef MATRIXOPTIMIZER_H
#define MATRIXOPTIMIZER_H
#include "Optimizer.h"
class MatrixOptimizer: public Optimizer {
public:
  virtual void setTask(TaskMatrix* task) = 0;
};
#endif //MATRIXOPTIMIZER_H
