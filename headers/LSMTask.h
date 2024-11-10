//
// Created by Eugene Bychkov on 03.11.2024.
//

#ifndef MINIMIZEROPTIMIZER_LSMTASK_H
#define MINIMIZEROPTIMIZER_LSMTASK_H

#include "TaskF.h"

class LSMTask: public Task{
    Function *c_function;
    std::vector<Function *> m_functions;
    std::vector<Variable *> m_X;
public:
    LSMTask(std::vector<Function *> functions, std::vector<Variable *> x) : m_functions(std::move(functions)), m_X(std::move(x)) {
        int i = 0;
        for (auto &function : m_functions){
            Constant *c = new Constant(2);
            Power *p = new Power(function, c);
            if (i == 0) {
                c_function = p;
            }else{
                c_function = new Addition(c_function, p);
            }
            i++;
        }
    }
    inline double getError() const override{
        return c_function->evaluate();
    }
    inline std::vector<double> getValues() const override{
        std::vector<double> values;
        for (auto & x : m_X) {
            values.push_back(x->evaluate());
        }
        return values;
    }
    double setError(std::vector<double> x) override{
        if (x.size() != m_X.size()) {
            throw std::invalid_argument("not right vector of variables");
        }
        for (int i = 0; i < x.size(); i++) {
            m_X[i]->setValue(x[i]);
        }
        return c_function->evaluate();
    }
    Matrix<> gradient() const override{
        Matrix<> grad(m_X.size(), 1);
        for (int i = 0; i < m_X.size(); i++) {
            grad(i, 0) = c_function->derivative(m_X[i])->evaluate();
        }
        return grad;
    }
    Matrix<> hessian() const override{
        Matrix<> hessian(m_X.size(), m_X.size());
        for (int i = 0; i < m_X.size(); i++) {
            for (int j = 0; j < m_X.size(); j++) {
                hessian(i, j) = c_function->derivative(m_X[i])->derivative(m_X[j])->evaluate();
            }
        }
        return hessian;
    }
    Matrix<> jacobian() const{
        Matrix<> jac(m_functions.size(), m_X.size());
        for (int i = 0; i < m_functions.size(); i++) {
            for (int j = 0; j < m_X.size(); j++) {
                jac(i, j) = 2 * m_functions[i]->evaluate() * m_functions[i]->derivative(m_X[j])->evaluate();
            }
        }
        return jac;
    }
};

#endif //MINIMIZEROPTIMIZER_LSMTASK_H
