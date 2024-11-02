//
// Created by Ardrass on 20.10.2024
//

#ifndef MINIMIZEROPTIMIZER_FUNCTION_H
#define MINIMIZEROPTIMIZER_FUNCTION_H

#include <cmath>
#include <stdexcept>

// Forward declaration
class Variable;

// Base abstract class Function
class Function {
public:
    virtual ~Function() = default;

    // Evaluate the function
    virtual double evaluate() const = 0;

    // Compute the derivative with respect to a specific variable
    virtual Function* derivative(Variable* var) const = 0;

    // Clone method
    virtual Function* clone() const = 0;
};

// Class Constant (constant function)
class Constant : public Function {
private:
    double value;

public:
    explicit Constant(double value);

    double evaluate() const override;

    Function* derivative(Variable* var) const override;

    Function* clone() const override;

    // Destructor (default is sufficient as there are no dynamic members)
    ~Constant() override = default;
};

// Class Variable
class Variable : public Function {
private:
    double* value; // Reference to the variable's value

public:
    explicit Variable(double* value);

    double evaluate() const override;

    Function* derivative(Variable* var) const override;

    Function* clone() const override;

    void setValue(double value);

    // Comparison operator to check if two Variables refer to the same double
    bool operator==(Variable* other) const;

};

// Class Addition
class Addition : public Function {
private:
    Function* left;
    Function* right;

public:
    Addition(Function* left, Function* right);

    double evaluate() const override;

    Function* derivative(Variable* var) const override;

    Function* clone() const override;


    // Disable copy constructor and copy assignment to prevent shallow copies
    Addition(const Addition&) = delete;
    Addition& operator=(const Addition&) = delete;
};

// Class Subtraction
class Subtraction : public Function {
private:
    Function* left;
    Function* right;

public:
    Subtraction(Function* left, Function* right);

    double evaluate() const override;

    Function* derivative(Variable* var) const override;

    Function* clone() const override;



    // Disable copy constructor and copy assignment to prevent shallow copies
    Subtraction(const Subtraction&) = delete;
    Subtraction& operator=(const Subtraction&) = delete;
};

// Class Multiplication
class Multiplication : public Function {
private:
    Function* left;
    Function* right;

public:
    Multiplication(Function* left, Function* right);

    double evaluate() const override;

    Function* derivative(Variable* var) const override;

    Function* clone() const override;


    // Disable copy constructor and copy assignment to prevent shallow copies
    Multiplication(const Multiplication&) = delete;
    Multiplication& operator=(const Multiplication&) = delete;
};

// Class Division
class Division : public Function {
private:
    Function* numerator;
    Function* denominator;

public:
    Division(Function* numerator, Function* denominator);

    double evaluate() const override;

    Function* derivative(Variable* var) const override;

    Function* clone() const override;



    // Disable copy constructor and copy assignment to prevent shallow copies
    Division(const Division&) = delete;
    Division& operator=(const Division&) = delete;
};

// Class Power
class Power : public Function {
private:
    Function* base;      // Base
    Function* exponent;  // Exponent

public:
    Power(Function* base, Function* exponent);

    double evaluate() const override;

    Function* derivative(Variable* var) const override;

    Function* clone() const override;



    // Disable copy constructor and copy assignment to prevent shallow copies
    Power(const Power&) = delete;
    Power& operator=(const Power&) = delete;
};

// Class Exponential (e^x)
class Exponential : public Function {
private:
    Function* exponent;

public:
    explicit Exponential(Function* exponent);

    double evaluate() const override;

    Function* derivative(Variable* var) const override;

    Function* clone() const override;



    // Disable copy constructor and copy assignment to prevent shallow copies
    Exponential(const Exponential&) = delete;
    Exponential& operator=(const Exponential&) = delete;
};

// Class Logarithm (ln(x))
class Logarithm : public Function {
private:
    Function* argument;

public:
    explicit Logarithm(Function* argument);

    double evaluate() const override;

    Function* derivative(Variable* var) const override;

    Function* clone() const override;



    // Disable copy constructor and copy assignment to prevent shallow copies
    Logarithm(const Logarithm&) = delete;
    Logarithm& operator=(const Logarithm&) = delete;
};

#endif // MINIMIZEROPTIMIZER_FUNCTION_H