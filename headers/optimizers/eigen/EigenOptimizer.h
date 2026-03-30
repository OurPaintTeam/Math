//
// Created by Eugene Bychkov on 18.05.2025.
//

#ifndef EIGENOPTIMIZER_H
#define EIGENOPTIMIZER_H
#include "Optimizer.h"
#include "TaskEigen.h"
class EigenOptimizer: public Optimizer {
public:
  virtual void setTask(TaskEigen* task) = 0;
};
#endif //EIGENOPTIMIZER_H
