#ifndef MINIMIZEROPTIMIZER_HEADERS_LSMFORLMTASK_H_
#define MINIMIZEROPTIMIZER_HEADERS_LSMFORLMTASK_H_
#include "TaskEigen.h"
#include <stdexcept>
#include <utility>

class LSMFORLMTask: public TaskEigen {
    Function* c_function;
    std::vector<Function*> m_functions;
    std::vector<Variable*> m_X;
    std::vector<std::vector<Function*>> m_jac;
public:
    LSMFORLMTask(std::vector<Function*> functions, std::vector<Variable*> x) : m_functions(functions), m_X(x) {
        c_function = nullptr;
        for (std::size_t i = 0; i < m_functions.size(); ++i) {
            Constant* two = new Constant(2.0);
            Power* squaredFunction = new Power(m_functions[i]->clone(), two);
            if (i == 0) {
                c_function = squaredFunction;
            } else {
                c_function = new Addition(c_function, squaredFunction);
            }
        }
        for (std::size_t j = 0; j < m_functions.size(); j++) {
            m_jac.push_back(std::vector<Function*>());
            for (std::size_t k = 0; k < m_X.size(); k++) {
                Function* dfdj = m_functions[j]->derivative(m_X[k]);
                m_jac[j].push_back(dfdj/*->simplify()*/);
                //delete dfdj;
            }
        }
    }

    ~LSMFORLMTask() {
        for (auto vec : m_jac) {
            for (auto f : vec) {
                delete f;
            }
        }
        delete c_function;
    }

    inline double getError() const override{
        return c_function->evaluate();
    }

    inline std::vector<double> getValues() const override{
        std::vector<double> values;
        values.reserve(m_X.size());
        for (const auto& x : m_X) {
            values.push_back(x->evaluate());
        }
        return values;
    }

    double setError(const std::vector<double>& x) override{
        if (x.size() != m_X.size()) {
            throw std::invalid_argument("Vector of variables has incorrect size.");
        }
        for (std::size_t i = 0; i < x.size(); ++i) {
            m_X[i]->setValue(x[i]);
        }
        return c_function->evaluate();
    }

    Eigen::VectorXd gradient() const override{
        Eigen::VectorXd grad(m_X.size());
        for (std::size_t i = 0; i < m_X.size(); ++i) {
            Function* partialDerivative = c_function->derivative(m_X[i]);
            grad(i) = partialDerivative->evaluate();
            delete partialDerivative;
        }
        return grad;
    }

    Eigen::MatrixXd hessian() const override{
        Eigen::MatrixXd hessianMatrix(m_X.size(), m_X.size());
        for (std::size_t i = 0; i < m_X.size(); ++i) {
            for (std::size_t j = 0; j < m_X.size(); ++j) {
                Function* secondPartialDerivative = c_function->derivative(m_X[i])->derivative(m_X[j]);
                hessianMatrix(i, j) = secondPartialDerivative->evaluate();
                delete secondPartialDerivative;
            }
        }
        return hessianMatrix;
    }

    Eigen::MatrixXd jacobian() const override{
        Eigen::MatrixXd jac(m_functions.size(), m_X.size());
        for (std::size_t i = 0; i < m_functions.size(); ++i) {
            for (std::size_t j = 0; j < m_X.size(); ++j) {
                Function* partialDerivative = m_functions[i]->derivative(m_X[j]);
                jac(i, j) = partialDerivative->evaluate();
                delete partialDerivative;
            }
        }
        return jac;
    }

    std::pair<Eigen::VectorXd, Eigen::MatrixXd> linearizeFunction() const override {
        Eigen::VectorXd residuals(m_functions.size());
        Eigen::MatrixXd jac(m_functions.size(), m_X.size());

        auto& functions = m_functions;
        auto& X = m_X;
        auto& jacobianFuncs = m_jac;

#pragma omp parallel for collapse(2) default(none) shared(functions, X, jacobianFuncs, residuals, jac)
        for (int i = 0; i < static_cast<int>(functions.size()); ++i) {
            for (int j = 0; j < static_cast<int>(X.size()); ++j) {
                if (j == 0) {
                    residuals(i) = functions[i]->evaluate();
                }
                jac(i, j) = jacobianFuncs[i][j]->evaluate();
            }
        }

        return { residuals, jac };
    }

    std::vector<Function*> getFunctions() const {
        return m_functions;
    }
};

#endif // ! MINIMIZEROPTIMIZER_HEADERS_LSMFORLMTASK_H_
