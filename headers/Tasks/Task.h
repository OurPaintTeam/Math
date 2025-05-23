#ifndef MINIMIZEROPTIMIZER_HEADERS_TASK_H_
#define MINIMIZEROPTIMIZER_HEADERS_TASK_H_

#include "Matrix.h"
#include "Function.h"
#include "ErrorFunction.h"
#include <vector>

class Task {
public:
    virtual ~Task() = default;
    virtual inline double getError() const =  0;
    virtual inline std::vector<double> getValues() const = 0;
    virtual double setError(const std::vector<double>& x) = 0;
};

#endif // ! MINIMIZEROPTIMIZER_HEADERS_TASK_H_
