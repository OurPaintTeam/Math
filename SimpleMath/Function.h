//
// Created by Eugene Bychkov on 19.10.2024.
//

#ifndef MINIMIZEROPTIMIZER_FUNCTION_H
#define MINIMIZEROPTIMIZER_FUNCTION_H

#include <vector>
#include "Matrix.h"

class Function {
protected:
    size_t numberOfVariables;
    std::vector<double> variables;
public:
    virtual double operator()() = 0;

    virtual std::vector<double> getVariable() = 0;

    virtual void setVariable(std::vector<double> variables) = 0;

    virtual size_t getNumberOfVariables() const = 0;

    virtual double getDerivative(size_t ID) const = 0;

    virtual double getMixedDerivative(size_t ID1, size_t ID2) const = 0;
};

class SquareFunction : public Function {
    size_t numberOfVariables;
    public:
    SquareFunction(const std::vector<double> &variables) : numberOfVariables(2) {
        this->variables = variables;
    }
    [[nodiscard]] size_t getNumberOfVariables() const override {
        return numberOfVariables;
    }

    double operator()() override {
        return variables[0] * variables[0] + variables[1] * variables[1];
    }

    std::vector<double> getVariable() override {
        return variables;
    }

    void setVariable(std::vector<double> variables) override {
        this->variables = variables;
    }
    double getDerivative(size_t ID) const override {
        if(ID == 0) {
            return 2 * variables[0];
        }else{
            return 2 * variables[1];
        }
    }
    double getMixedDerivative(size_t ID1, size_t ID2) const override {
        if(ID1 == ID2) {
            return 2;
        }
        return 0;
    }

};

#endif //MINIMIZEROPTIMIZER_FUNCTION_H
