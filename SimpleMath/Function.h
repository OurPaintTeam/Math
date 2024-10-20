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
    Function(const std::vector<double> & var): variables(var), numberOfVariables(var.size()){}
    virtual double operator()() = 0;

    virtual std::vector<double> getVariable() {
        return variables;
    };

    virtual void setVariable(std::vector<double> variables) {
        this->variables = variables;
    };

    virtual size_t getNumberOfVariables() const{
        return numberOfVariables;
    };

    virtual double getDerivative(size_t ID) const = 0;

    virtual double getMixedDerivative(size_t ID1, size_t ID2) const = 0;
};

class SquareFunction : public Function {
    public:
    SquareFunction(const std::vector<double> &variables): Function(variables){}

    double operator()() override {
        double sum = 0;
        for(size_t i = 0; i < numberOfVariables; i++) {
            sum += variables[i] * variables[i];
        }
        return sum;
    }
    double getDerivative(size_t ID) const override {
        return 2*variables[ID];
    }
    double getMixedDerivative(size_t ID1, size_t ID2) const override {
        if(ID1 == ID2) {
            return 2;
        }
        return 0;
    }

};

#endif //MINIMIZEROPTIMIZER_FUNCTION_H
