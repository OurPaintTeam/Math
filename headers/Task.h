//
// Created by Eugene Bychkov on 03.11.2024.
//

#ifndef MINIMIZEROPTIMIZER_TASK_H
#define MINIMIZEROPTIMIZER_TASK_H
class Task {
    public:
    virtual Matrix<> gradient() const = 0;
    virtual Matrix<> hessian() const = 0;
    virtual inline double getError() const =  0;
    virtual inline std::vector<double> getValues() const = 0;
    virtual double setError(const std::vector<double> & x) = 0;
};
#endif //MINIMIZEROPTIMIZER_TASK_H
