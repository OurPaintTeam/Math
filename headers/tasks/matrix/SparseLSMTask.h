#ifndef MINIMIZEROPTIMIZER_HEADERS_TASKS_MATRIX_SPARSELSMTASK_H_
#define MINIMIZEROPTIMIZER_HEADERS_TASKS_MATRIX_SPARSELSMTASK_H_

#include "SparseMatrix.h"
#include "TaskMatrix.h"

#include <algorithm>
#include <functional>
#include <stdexcept>
#include <utility>
#include <vector>

class SparseLSMTask : public TaskMatrix {
public:
    using LinearizationView =
        std::pair<std::reference_wrapper<const Matrix<>>,
                  std::reference_wrapper<const SparseMatrix<>>>;

private:
    struct JacobianEntry {
        size_t residualIndex = 0;
        size_t variableIndex = 0;
        double* jacobianValue = nullptr;
        Function* derivative = nullptr;
    };

    struct HessianContribution {
        const double* leftJacobianValue = nullptr;
        const double* rightJacobianValue = nullptr;
        double* hessianValue = nullptr;
    };

    std::vector<Function*> m_functions;
    std::vector<Variable*> m_X;
    std::vector<JacobianEntry> m_jacobianEntries;
    std::vector<size_t> m_rowEntryOffsets;
    std::vector<HessianContribution> m_hessianContributions;
    std::vector<double*> m_hessianValueRefs;
    std::vector<size_t> m_hessianDiagonalPositions;

    mutable Matrix<> m_residualVector;
    mutable SparseMatrix<> m_jacobian;
    mutable Matrix<> m_normalGradient;
    mutable Matrix<> m_objectiveGradient;
    mutable SparseMatrix<> m_approximateHessian;
    mutable Matrix<> m_denseObjectiveHessian;
    mutable double m_error = 0.0;
    mutable std::vector<double> m_cachedVariableValues;
    mutable bool m_linearizationDirty = true;
    mutable bool m_objectiveGradientDirty = true;
    mutable bool m_approximateHessianDirty = true;
    mutable bool m_denseObjectiveHessianDirty = true;

    static bool isZeroFunction(const Function* function) {
        const auto* constant = dynamic_cast<const Constant*>(function);
        return constant != nullptr && constant->evaluate() == 0.0;
    }

    void buildSymbolicJacobian() {
        m_jacobianEntries.clear();
        m_jacobianEntries.reserve(m_functions.size() * m_X.size());

        m_rowEntryOffsets.clear();
        m_rowEntryOffsets.reserve(m_functions.size() + 1);
        m_rowEntryOffsets.push_back(0);

        std::vector<size_t> columnCounts(m_X.size(), 0);

        for (size_t residualIndex = 0; residualIndex < m_functions.size(); ++residualIndex) {
            for (size_t variableIndex = 0; variableIndex < m_X.size(); ++variableIndex) {
                Function* derivative = m_functions[residualIndex]->derivative(m_X[variableIndex]);
                Function* simplified = derivative->simplify();
                delete derivative;

                if (isZeroFunction(simplified)) {
                    delete simplified;
                    continue;
                }

                m_jacobianEntries.push_back({
                    residualIndex,
                    variableIndex,
                    nullptr,
                    simplified
                });
                ++columnCounts[variableIndex];
            }
            m_rowEntryOffsets.push_back(m_jacobianEntries.size());
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
            inner[position] = entry.residualIndex;
        }

        m_jacobian = SparseMatrix<>::fromCSC(
            m_functions.size(),
            m_X.size(),
            std::move(values),
            std::move(inner),
            std::move(outer));

        for (JacobianEntry& entry : m_jacobianEntries) {
            entry.jacobianValue = &m_jacobian.coeffRef(entry.residualIndex, entry.variableIndex);
        }
    }

    void buildSymbolicHessian() {
        const size_t variableCount = m_X.size();
        std::vector<std::vector<size_t>> columnRows(variableCount);
        std::vector<size_t> columnValueOffsets(variableCount, 0);
        for (size_t col = 0; col < variableCount; ++col) {
            columnRows[col].push_back(col);
        }

        size_t contributionCount = 0;
        for (size_t residualIndex = 0; residualIndex < m_functions.size(); ++residualIndex) {
            const size_t begin = m_rowEntryOffsets[residualIndex];
            const size_t end = m_rowEntryOffsets[residualIndex + 1];
            const size_t rowNnz = end - begin;
            contributionCount += rowNnz * rowNnz;

            for (size_t right = begin; right < end; ++right) {
                auto& rows = columnRows[m_jacobianEntries[right].variableIndex];
                for (size_t left = begin; left < end; ++left) {
                    rows.push_back(m_jacobianEntries[left].variableIndex);
                }
            }
        }

        std::vector<size_t> outer(variableCount + 1, 0);
        std::vector<size_t> inner;
        inner.reserve(variableCount == 0 ? 0 : contributionCount + variableCount);
        std::vector<double> values;
        values.reserve(variableCount == 0 ? 0 : contributionCount + variableCount);

        m_hessianValueRefs.clear();
        m_hessianValueRefs.reserve(variableCount == 0 ? 0 : contributionCount + variableCount);

        m_hessianDiagonalPositions.clear();
        m_hessianDiagonalPositions.reserve(variableCount);

        for (size_t col = 0; col < variableCount; ++col) {
            auto& rows = columnRows[col];
            std::sort(rows.begin(), rows.end());
            rows.erase(std::unique(rows.begin(), rows.end()), rows.end());

            outer[col] = inner.size();
            columnValueOffsets[col] = values.size();
            for (size_t row : rows) {
                if (row == col) {
                    m_hessianDiagonalPositions.push_back(values.size());
                }
                inner.push_back(row);
                values.push_back(0.0);
            }
        }
        outer[variableCount] = inner.size();

        m_approximateHessian = SparseMatrix<>::fromCSC(
            variableCount,
            variableCount,
            std::move(values),
            std::move(inner),
            std::move(outer));

        m_hessianContributions.clear();
        m_hessianContributions.reserve(contributionCount);
        for (size_t col = 0; col < variableCount; ++col) {
            const auto& rows = columnRows[col];
            for (size_t idx = 0; idx < rows.size(); ++idx) {
                m_hessianValueRefs.push_back(&m_approximateHessian.coeffRef(rows[idx], col));
            }
        }

        for (size_t residualIndex = 0; residualIndex < m_functions.size(); ++residualIndex) {
            const size_t begin = m_rowEntryOffsets[residualIndex];
            const size_t end = m_rowEntryOffsets[residualIndex + 1];
            for (size_t left = begin; left < end; ++left) {
                for (size_t right = begin; right < end; ++right) {
                    const size_t col = m_jacobianEntries[right].variableIndex;
                    const auto& rows = columnRows[col];
                    const auto rowIt = std::lower_bound(
                        rows.begin(),
                        rows.end(),
                        m_jacobianEntries[left].variableIndex);
                    const size_t rowOffset = static_cast<size_t>(rowIt - rows.begin());
                    const size_t valuePosition = columnValueOffsets[col] + rowOffset;
                    m_hessianContributions.push_back({
                        m_jacobianEntries[left].jacobianValue,
                        m_jacobianEntries[right].jacobianValue,
                        m_hessianValueRefs[valuePosition]
                    });
                }
            }
        }
    }

    void invalidateNumericCaches() const {
        m_linearizationDirty = true;
        m_objectiveGradientDirty = true;
        m_approximateHessianDirty = true;
        m_denseObjectiveHessianDirty = true;
    }

    void refreshDirtyState() const {
        bool changed = false;
        for (size_t i = 0; i < m_X.size(); ++i) {
            const double currentValue = m_X[i]->evaluate();
            if (currentValue != m_cachedVariableValues[i]) {
                m_cachedVariableValues[i] = currentValue;
                changed = true;
            }
        }

        if (changed) {
            invalidateNumericCaches();
        }
    }

    void ensureLinearization() const {
        refreshDirtyState();
        if (!m_linearizationDirty) {
            return;
        }

        for (size_t i = 0; i < m_X.size(); ++i) {
            m_normalGradient(i, 0) = 0.0;
        }

        double error = 0.0;
        for (size_t residualIndex = 0; residualIndex < m_functions.size(); ++residualIndex) {
            const double residual = m_functions[residualIndex]->evaluate();
            m_residualVector(residualIndex, 0) = residual;
            error += residual * residual;

            const size_t begin = m_rowEntryOffsets[residualIndex];
            const size_t end = m_rowEntryOffsets[residualIndex + 1];
            for (size_t entryIndex = begin; entryIndex < end; ++entryIndex) {
                const JacobianEntry& entry = m_jacobianEntries[entryIndex];
                const double value = entry.derivative->evaluate();
                *entry.jacobianValue = value;
                m_normalGradient(entry.variableIndex, 0) += value * residual;
            }
        }

        m_error = error;
        m_linearizationDirty = false;
        m_objectiveGradientDirty = true;
        m_approximateHessianDirty = true;
        m_denseObjectiveHessianDirty = true;
    }

    void ensureObjectiveGradient() const {
        ensureLinearization();
        if (!m_objectiveGradientDirty) {
            return;
        }

        for (size_t i = 0; i < m_X.size(); ++i) {
            m_objectiveGradient(i, 0) = 2.0 * m_normalGradient(i, 0);
        }
        m_objectiveGradientDirty = false;
    }

    void ensureApproximateHessian() const {
        ensureLinearization();
        if (!m_approximateHessianDirty) {
            return;
        }

        for (double* valueRef : m_hessianValueRefs) {
            *valueRef = 0.0;
        }

        for (const HessianContribution& contribution : m_hessianContributions) {
            const double value =
                (*contribution.leftJacobianValue) * (*contribution.rightJacobianValue);
            *contribution.hessianValue += value;
        }

        m_approximateHessianDirty = false;
        m_denseObjectiveHessianDirty = true;
    }

    void ensureDenseObjectiveHessian() const {
        ensureApproximateHessian();
        if (!m_denseObjectiveHessianDirty) {
            return;
        }

        m_denseObjectiveHessian.setZeroes();
        const auto& outer = m_approximateHessian.outerIndexData();
        const auto& inner = m_approximateHessian.innerIndexData();
        const auto& values = m_approximateHessian.valueData();

        for (size_t col = 0; col < m_approximateHessian.cols_size(); ++col) {
            for (size_t pos = outer[col]; pos < outer[col + 1]; ++pos) {
                m_denseObjectiveHessian(inner[pos], col) = 2.0 * values[pos];
            }
        }

        m_denseObjectiveHessianDirty = false;
    }

public:
    SparseLSMTask(std::vector<Function*> functions, std::vector<Variable*> x)
        : m_functions(std::move(functions)),
          m_X(std::move(x)),
          m_residualVector(m_functions.size(), 1),
          m_jacobian(m_functions.size(), m_X.size()),
          m_normalGradient(m_X.size(), 1),
          m_objectiveGradient(m_X.size(), 1),
          m_approximateHessian(m_X.size(), m_X.size()),
          m_denseObjectiveHessian(m_X.size(), m_X.size()),
          m_cachedVariableValues(m_X.size(), 0.0)
    {
        buildSymbolicJacobian();
        buildSymbolicHessian();

        for (size_t i = 0; i < m_X.size(); ++i) {
            m_cachedVariableValues[i] = m_X[i]->evaluate();
        }
    }

    ~SparseLSMTask() override {
        for (Function* function : m_functions) {
            if (function != nullptr && function->getType() != VARIABLE) {
                delete function;
            }
        }

        for (const JacobianEntry& entry : m_jacobianEntries) {
            delete entry.derivative;
        }
    }

    Matrix<> residuals() const {
        ensureLinearization();
        return m_residualVector;
    }

    const Matrix<>& residualsRef() const {
        ensureLinearization();
        return m_residualVector;
    }

    SparseMatrix<> jacobian() const {
        ensureLinearization();
        return m_jacobian;
    }

    const SparseMatrix<>& jacobianRef() const {
        ensureLinearization();
        return m_jacobian;
    }

    const Matrix<>& normalGradient() const {
        ensureLinearization();
        return m_normalGradient;
    }

    const SparseMatrix<>& approximateHessian() const {
        ensureApproximateHessian();
        return m_approximateHessian;
    }

    SparseMatrix<> dampedNormalMatrix(double lambda) const {
        ensureApproximateHessian();
        std::vector<double> values = m_approximateHessian.valueData();
        for (size_t position : m_hessianDiagonalPositions) {
            values[position] += lambda;
        }

        return SparseMatrix<>::fromCSC(
            m_approximateHessian.rows_size(),
            m_approximateHessian.cols_size(),
            std::move(values),
            std::vector<size_t>(m_approximateHessian.innerIndexData()),
            std::vector<size_t>(m_approximateHessian.outerIndexData()));
    }

    LinearizationView linearizationView() const {
        ensureLinearization();
        return {std::cref(m_residualVector), std::cref(m_jacobian)};
    }

    std::pair<Matrix<>, SparseMatrix<>> linearizeFunction() const {
        ensureLinearization();
        return {m_residualVector, m_jacobian};
    }

    double getError() const override {
        ensureLinearization();
        return m_error;
    }

    std::vector<double> getValues() const override {
        std::vector<double> values;
        values.reserve(m_X.size());
        for (Variable* variable : m_X) {
            values.push_back(variable->evaluate());
        }
        return values;
    }

    double setError(const std::vector<double>& x) override {
        if (x.size() != m_X.size()) {
            throw std::invalid_argument("Vector of variables has incorrect size.");
        }

        for (size_t i = 0; i < x.size(); ++i) {
            m_X[i]->setValue(x[i]);
            m_cachedVariableValues[i] = x[i];
        }

        invalidateNumericCaches();
        return getError();
    }

    Matrix<> gradient() const override {
        ensureObjectiveGradient();
        return m_objectiveGradient;
    }

    Matrix<> hessian() const override {
        ensureDenseObjectiveHessian();
        return m_denseObjectiveHessian;
    }
};

#endif // MINIMIZEROPTIMIZER_HEADERS_TASKS_MATRIX_SPARSELSMTASK_H_
