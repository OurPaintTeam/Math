#ifndef MINIMIZEROPTIMIZER_HEADERS_LSMFORLMTASK_H_
#define MINIMIZEROPTIMIZER_HEADERS_LSMFORLMTASK_H_

#include "TaskF.h"
#include <Eigen/Dense>
#include <stdexcept>
#include <utility>

class LSMFORLMTask {
    Function* c_function;
    std::vector<Function*> m_functions;
    std::vector<Variable*> m_X;

public:
    LSMFORLMTask(std::vector<Function*> functions, std::vector<Variable*> x) : m_functions(functions), m_X(x) {
        c_function = nullptr;
        for (size_t i = 0; i < m_functions.size(); ++i) {
            Constant* two = new Constant(2.0);
            Power* squaredFunction = new Power(m_functions[i]->clone(), two);
            if (i == 0) {
                c_function = squaredFunction;
            } else {
                c_function = new Addition(c_function, squaredFunction);
            }
        }
    }

    ~LSMFORLMTask() {
        for (Function* f : m_functions) {
            delete f;
        }
        delete c_function;
    }

    inline double getError() const {
        return c_function->evaluate();
    }

    inline std::vector<double> getValues() const {
        std::vector<double> values;
        values.reserve(m_X.size());
        for (const auto& x : m_X) {
            values.push_back(x->evaluate());
        }
        return values;
    }

    double setError(const std::vector<double>& x) {
        if (x.size() != m_X.size()) {
            throw std::invalid_argument("Vector of variables has incorrect size.");
        }
        for (size_t i = 0; i < x.size(); ++i) {
            m_X[i]->setValue(x[i]);
        }
        return c_function->evaluate();
    }

    Eigen::VectorXd gradient() const {
        Eigen::VectorXd grad(m_X.size());
        for (size_t i = 0; i < m_X.size(); ++i) {
            Function* partialDerivative = c_function->derivative(m_X[i]);
            grad(i) = partialDerivative->evaluate();
            delete partialDerivative;
        }
        return grad;
    }

    Eigen::MatrixXd hessian() const {
        Eigen::MatrixXd hessianMatrix(m_X.size(), m_X.size());
        for (size_t i = 0; i < m_X.size(); ++i) {
            for (size_t j = 0; j < m_X.size(); ++j) {
                Function* secondPartialDerivative = c_function->derivative(m_X[i])->derivative(m_X[j]);
                hessianMatrix(i, j) = secondPartialDerivative->evaluate();
                delete secondPartialDerivative;
            }
        }
        return hessianMatrix;
    }

    Eigen::MatrixXd jacobian() const {
        Eigen::MatrixXd jac(m_functions.size(), m_X.size());
        for (size_t i = 0; i < m_functions.size(); ++i) {
            for (size_t j = 0; j < m_X.size(); ++j) {
                Function* partialDerivative = m_functions[i]->derivative(m_X[j]);
                jac(i, j) = partialDerivative->evaluate();
                delete partialDerivative;
            }
        }
        return jac;
    }

    std::pair<Eigen::VectorXd, Eigen::MatrixXd> linearizeFunction() const {
        Eigen::VectorXd residuals(m_functions.size());
        Eigen::MatrixXd jac(m_functions.size(), m_X.size());

        for (size_t i = 0; i < m_functions.size(); ++i) {
            residuals(i) = m_functions[i]->evaluate();
            for (size_t j = 0; j < m_X.size(); ++j) {
                Function* partialDerivative = m_functions[i]->derivative(m_X[j]);
                jac(i, j) = partialDerivative->evaluate();
                delete partialDerivative;
            }
        }
        return {residuals, jac};
    }

    std::vector<Function*> getFunctions() const {
        return m_functions;
    }
};

#endif // ! MINIMIZEROPTIMIZER_HEADERS_LSMFORLMTASK_H_
