#ifndef MINIMIZEROPTIMIZER_HEADERS_TASKS_MATRIX_SPARSESYSTEMTASK_H_
#define MINIMIZEROPTIMIZER_HEADERS_TASKS_MATRIX_SPARSESYSTEMTASK_H_

#include "SparseMatrix.h"
#include "TaskMatrix.h"

#include <algorithm>
#include <stdexcept>
#include <utility>
#include <vector>

class SparseSystemTask : public TaskMatrix {
private:
    struct JacobianEntry {
        size_t functionIndex = 0;
        size_t variableIndex = 0;
        double* jacobianValue = nullptr;
        Function* derivative = nullptr;
    };

    std::vector<Function*> m_functions;
    std::vector<Variable*> m_X;
    std::vector<JacobianEntry> m_jacobianEntries;

    mutable Matrix<> m_values;
    mutable SparseMatrix<> m_jacobian;
    mutable double m_error = 0.0;
    mutable bool m_linearizationDirty = true;

    static bool isZeroFunction(const Function* function) {
        const auto* constant = dynamic_cast<const Constant*>(function);
        return constant != nullptr && constant->evaluate() == 0.0;
    }

    void buildSymbolicJacobian() {
        m_jacobianEntries.clear();
        m_jacobianEntries.reserve(m_functions.size() * m_X.size());

        std::vector<size_t> columnCounts(m_X.size(), 0);
        for (size_t functionIndex = 0; functionIndex < m_functions.size(); ++functionIndex) {
            for (size_t variableIndex = 0; variableIndex < m_X.size(); ++variableIndex) {
                Function* derivative = m_functions[functionIndex]->derivative(m_X[variableIndex]);
                Function* simplified = derivative->simplify();
                delete derivative;

                if (isZeroFunction(simplified)) {
                    delete simplified;
                    continue;
                }

                m_jacobianEntries.push_back({
                    functionIndex,
                    variableIndex,
                    nullptr,
                    simplified
                });
                ++columnCounts[variableIndex];
            }
        }

        std::vector<size_t> outer(m_X.size() + 1, 0);
        for (size_t col = 0; col < m_X.size(); ++col) {
            outer[col + 1] = outer[col] + columnCounts[col];
        }

        std::vector<size_t> next(outer);
        std::vector<size_t> inner(m_jacobianEntries.size(), 0);
        std::vector<double> values(m_jacobianEntries.size(), 0.0);
        for (JacobianEntry& entry : m_jacobianEntries) {
            const size_t position = next[entry.variableIndex]++;
            inner[position] = entry.functionIndex;
        }

        m_jacobian = SparseMatrix<>::fromCSC(
            m_functions.size(),
            m_X.size(),
            std::move(values),
            std::move(inner),
            std::move(outer));

        for (JacobianEntry& entry : m_jacobianEntries) {
            entry.jacobianValue = &m_jacobian.coeffRef(entry.functionIndex, entry.variableIndex);
        }
    }

    void ensureLinearization() const {
        if (!m_linearizationDirty) {
            return;
        }

        m_error = 0.0;
        for (size_t i = 0; i < m_functions.size(); ++i) {
            const double value = m_functions[i]->evaluate();
            m_values(i, 0) = value;
            m_error += value * value;
        }

        for (const JacobianEntry& entry : m_jacobianEntries) {
            *entry.jacobianValue = entry.derivative->evaluate();
        }

        m_linearizationDirty = false;
    }

public:
    SparseSystemTask(std::vector<Function*> functions, std::vector<Variable*> x)
        : m_functions(std::move(functions)),
          m_X(std::move(x)),
          m_values(m_functions.size(), 1),
          m_jacobian(m_functions.size(), m_X.size())
    {
        buildSymbolicJacobian();
    }

    ~SparseSystemTask() override {
        for (Function* function : m_functions) {
            if (function != nullptr && function->getType() != VARIABLE) {
                delete function;
            }
        }

        for (const JacobianEntry& entry : m_jacobianEntries) {
            delete entry.derivative;
        }
    }

    Matrix<> values() const {
        ensureLinearization();
        return m_values;
    }

    const Matrix<>& valuesRef() const {
        ensureLinearization();
        return m_values;
    }

    SparseMatrix<> jacobian() const {
        ensureLinearization();
        return m_jacobian;
    }

    const SparseMatrix<>& jacobianRef() const {
        ensureLinearization();
        return m_jacobian;
    }

    void fillJacobian(SparseMatrix<>& target) const {
        ensureLinearization();
        const bool samePattern =
            !target.rowMutableMode()
            && target.rows_size() == m_jacobian.rows_size()
            && target.cols_size() == m_jacobian.cols_size()
            && target.innerIndexData() == m_jacobian.innerIndexData()
            && target.outerIndexData() == m_jacobian.outerIndexData();

        if (!samePattern) {
            target = m_jacobian;
            return;
        }

        target.mutableValueData() = m_jacobian.valueData();
    }

    Matrix<> gradient() const override {
        ensureLinearization();
        return m_values;
    }

    Matrix<> hessian() const override {
        ensureLinearization();
        Matrix<> denseJacobian(m_jacobian.rows_size(), m_jacobian.cols_size());
        const auto& outer = m_jacobian.outerIndexData();
        const auto& inner = m_jacobian.innerIndexData();
        const auto& values = m_jacobian.valueData();
        for (size_t col = 0; col < m_jacobian.cols_size(); ++col) {
            for (size_t pos = outer[col]; pos < outer[col + 1]; ++pos) {
                denseJacobian(inner[pos], col) = values[pos];
            }
        }
        return denseJacobian;
    }

    double getError() const override {
        ensureLinearization();
        return m_error;
    }

    std::vector<double> getValues() const override {
        std::vector<double> values;
        values.reserve(m_X.size());
        for (Variable* x : m_X) {
            values.push_back(x->evaluate());
        }
        return values;
    }

    double setError(const std::vector<double>& x) override {
        if (x.size() != m_X.size()) {
            throw std::invalid_argument("Values size must match variables count");
        }

        for (size_t i = 0; i < m_X.size(); ++i) {
            m_X[i]->setValue(x[i]);
        }
        m_linearizationDirty = true;
        ensureLinearization();
        return m_error;
    }
};

#endif // MINIMIZEROPTIMIZER_HEADERS_TASKS_MATRIX_SPARSESYSTEMTASK_H_
