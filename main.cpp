//
// Created by Eugene Bychkov on 19.10.2024.
//
#include "MinimizeTask.h"

int main() {
    Function *function = new SquareFunction({2000, 1000});
    MinimizeTask task(function);
    task.getHessian().print();
    return 0;
}