#ifndef MINIMIZEROPTIMIZER_TASKF_H_
#define MINIMIZEROPTIMIZER_TASKF_H_

#include "Function.h"
#include "Matrix.h"
#include "TaskMatrix.h"

class TaskF: public TaskMatrix {
    Function* c_function;
    std::vector<Variable*> m_X;
    std::vector<Function*> m_grad;
    std::vector<std::vector<Function*>> m_hess; // no constructor from non-Arithmetic(Function*) arguments for matrix

public:
    TaskF(Function* c_function, std::vector<Variable* > x): c_function(c_function), m_X(x){
        for (std::size_t i = 0; i < m_X.size(); i++) {
            m_grad.push_back(c_function->derivative(m_X[i]));
        }
        for (std::size_t i = 0; i < m_X.size(); i++) {
            m_hess.push_back(std::vector<Function*>());
            for (std::size_t j = 0; j < m_X.size(); j++) {
                m_hess[i].push_back(m_grad[i]->derivative(m_X[j]));
            }
        }
    }

    ~TaskF() {
        if (c_function->getType() == VARIABLE) {
            return;
        }
	    delete c_function;
	    for (Function* f : m_grad) {
		    delete f;
	    }
	    for (auto& row : m_hess) {
		    for (auto& func : row) {
			    delete func;
		    }
	    }
    }

    Matrix<> gradient() const override {
        Matrix<> gradient(m_X.size(), 1);
        for (std::size_t i = 0; i < m_X.size(); i++) {
            gradient(i, 0) = m_grad[i]->evaluate();
        }
        return gradient;
    }

    Matrix<> hessian() const override{
        Matrix<> hessian(m_X.size(), m_X.size());
        for (std::size_t i = 0; i < m_X.size(); i++) {
            for (std::size_t j = 0; j < m_X.size(); j++) {
                hessian(i, j) = m_hess[i][j]->evaluate();
            }
        }
        return hessian;
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

    double setError(const std::vector<double> & x) override{
        for (std::size_t i = 0; i < m_X.size(); i++) {
            m_X[i]->setValue(x[i]);
        }
        return c_function->evaluate();
    }
};
#endif // ! MINIMIZEROPTIMIZER_TASKF_H_
