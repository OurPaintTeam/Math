#ifndef MINIMIZEROPTIMIZER_HEADERS_LSMTASK_H_
#define MINIMIZEROPTIMIZER_HEADERS_LSMTASK_H_

#include "TaskF.h"

class LSMTask : public Task {
    Function *c_function;
    std::vector<Function *> m_functions;
    std::vector<Variable *> m_X;
    std::vector<Function *> m_grad;
    std::vector<std::vector<Function *> > m_jac;
    std::vector<std::vector<Function *> > m_hess;

public:
    LSMTask(std::vector<Function *> functions, std::vector<Variable *> x) : m_functions(std::move(functions)),
                                                                            m_X(std::move(x)) {
        int i = 0;
        for (auto &function: m_functions) {
            Constant *c = new Constant(2);
            Power *p = new Power(function, c);
            if (i == 0) {
                c_function = p;
            } else {
                c_function = new Addition(c_function, p);
            }
            i++;
        }
        for (int j = 0; j < m_X.size(); j++) {
            m_grad.push_back(c_function->derivative(m_X[j]));
        }
        for (int j = 0; j < m_X.size(); j++) {
            m_hess.push_back(std::vector<Function *>());
            for (int k = 0; k < m_X.size(); k++) {
                m_hess[j].push_back(c_function->derivative(m_X[j])->derivative(m_X[k]));
            }
        }
        for (int j = 0; j < m_functions.size(); j++) {
            m_jac.push_back(std::vector<Function *>());
            for (int k = 0; k < m_X.size(); k++) {
                m_jac[j].push_back(m_functions[j]->derivative(m_X[k]));
            }
        }
    }

    inline double getError() const override {
        return c_function->evaluate();
    }

    inline std::vector<double> getValues() const override {
        std::vector<double> values;
        for (auto &x: m_X) {
            values.push_back(x->evaluate());
        }
        return values;
    }

    double setError(const std::vector<double> &x) override {
        if (x.size() != m_X.size()) {
            throw std::invalid_argument("not right vector of variables");
        }
        for (int i = 0; i < x.size(); i++) {
            m_X[i]->setValue(x[i]);
        }
        return c_function->evaluate();
    }

    Matrix<> gradient() const override {
        Matrix<> grad(m_X.size(), 1);
        for (int i = 0; i < m_X.size(); i++) {
            grad(i, 0) = m_grad[i]->evaluate();
        }
        return grad;
    }

    Matrix<> hessian() const override {
        Matrix<> hessian(m_X.size(), m_X.size());
        for (int i = 0; i < m_X.size(); i++) {
            for (int j = 0; j < m_X.size(); j++) {
                hessian(i, j) = m_hess[i][j]->evaluate();
            }
        }
        return hessian;
    }

    Matrix<> jacobian() const {
        Matrix<> jac(m_functions.size(), m_X.size());
        for (int i = 0; i < m_functions.size(); i++) {
            for (int j = 0; j < m_X.size(); j++) {
                jac(i, j) = m_jac[i][j]->evaluate();
            }
        }
        return jac;
    }

    std::pair<Matrix<>, Matrix<> > linearizeFunction() const {
        Matrix<> residuals(m_functions.size(), 1);
        Matrix<> jac(m_functions.size(), m_X.size());

        for (int i = 0; i < m_functions.size(); ++i) {
            residuals(i, 0) = m_functions[i]->evaluate();
            for (int j = 0; j < m_X.size(); ++j) {
                jac(i, j) = m_jac[i][j]->evaluate();
            }
        }
        return {residuals, jac};
    }

    ~LSMTask() {
        delete c_function;
        for (auto func: m_functions) {
            delete func;
        }
        for (auto func: m_grad) {
            delete func;
        }
        for (auto func: m_hess) {
            for (auto f: func) {
                delete f;
            }
        }
        for (auto func: m_jac) {
            for (auto f: func) {
                delete f;
            }
        }
    }
};

#endif // ! MINIMIZEROPTIMIZER_HEADERS_LSMTASK_H_
