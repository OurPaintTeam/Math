#ifndef MINIMIZEROPTIMIZER_FUNCTION_H
#define MINIMIZEROPTIMIZER_FUNCTION_H

#include <cmath>
#include <stdexcept>



// Function

//  Constant
//  Variable
//  Addition
//  Subtraction
//  Multiplication
//  Division
//  Power
//  Negation
//  Abs
//  Sign
//  Modulo
//  Exp
//  Ln
//  Log
//  Sqrt
//  Sin
//  Cos
//  Asin
//  Acos
//  Tan
//  Atan






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

// Class Negation
class Negation : public Function {
private:
    Function* argument;

public:
    explicit Negation(Function* argument);

    double evaluate() const override;

    Function* derivative(Variable* var) const override;

    Function* clone() const override;



    // Disable copy constructor and copy assignment to prevent shallow copies
    Negation(const Negation&) = delete;
    Negation& operator=(const Negation&) = delete;
};

// Class Abs
class Abs : public Function {
private:
    Function* argument;

public:
    explicit Abs(Function* argument);

    double evaluate() const override;

    Function* derivative(Variable* var) const override;

    Function* clone() const override;



    // Disable copy constructor and copy assignment to prevent shallow copies
    Abs(const Abs&) = delete;
    Abs& operator=(const Abs&) = delete;
};

// Class Sign
class Sign : public Function {
private:
    Function* argument;

public:
    explicit Sign(Function* argument);

    double evaluate() const override;

    Function* derivative(Variable* var) const override;

    Function* clone() const override;



    // Disable copy constructor and copy assignment to prevent shallow copies
    Sign(const Sign&) = delete;
    Sign& operator=(const Sign&) = delete;
};

// Class Modulo
class Mod : public Function {
private:
    Function* numerator;
    Function* denominator;

public:
    explicit Mod(Function* numerator, Function* denominator);

    double evaluate() const override;

    Function* derivative(Variable* var) const override;

    Function* clone() const override;



    // Disable copy constructor and copy assignment to prevent shallow copies
    Mod(const Mod&) = delete;
    Mod& operator=(const Mod&) = delete;
};

// Class Exponential (e^x)
class Exp : public Function {
private:
    Function* exponent;

public:
    explicit Exp(Function* exponent);

    double evaluate() const override;

    Function* derivative(Variable* var) const override;

    Function* clone() const override;



    // Disable copy constructor and copy assignment to prevent shallow copies
    Exp(const Exp&) = delete;
    Exp& operator=(const Exp&) = delete;
};

// Class Ln (ln(x))
class Ln : public Function {
private:
    Function* argument;

public:
    explicit Ln(Function* argument);

    double evaluate() const override;

    Function* derivative(Variable* var) const override;

    Function* clone() const override;



    // Disable copy constructor and copy assignment to prevent shallow copies
    Ln(const Ln&) = delete;
    Ln& operator=(const Ln&) = delete;
};

// Class Log (log(base, x))
class Log : public Function {
private:
    Function* base;
    Function* argument;

public:
    explicit Log(Function* base, Function* argument);

    double evaluate() const override;

    Function* derivative(Variable* var) const override;

    Function* clone() const override;



    // Disable copy constructor and copy assignment to prevent shallow copies
    Log(const Ln&) = delete;
    Log& operator=(const Log&) = delete;
};

// class sqrt (sqrt(x))
class Sqrt : public Function {
private:
    Function* argument;

public:
    explicit Sqrt(Function* argument);

    double evaluate() const override;

    Function* derivative(Variable* var) const override;

    Function* clone() const override;



    // Disable copy constructor and copy assignment to prevent shallow copies
    Sqrt(const Sqrt&) = delete;
    Sqrt& operator=(const Sqrt&) = delete;
};

// Class sin (sin(x))
class Sin : public Function {
private:
    Function* argument;

public:
    explicit Sin(Function* argument);

    double evaluate() const override;

    Function* derivative(Variable* var) const override;

    Function* clone() const override;

    // Disable copy constructor and copy assignment to prevent shallow copies
    Sin(const Sin&) = delete;
    Sin& operator=(const Sin&) = delete;
};

// Class cos (cos(x))
class Cos : public Function {
private:
    Function* argument;

public:
    explicit Cos(Function* argument);

    double evaluate() const override;

    Function* derivative(Variable* var) const override;

    Function* clone() const override;

    // Disable copy constructor and copy assignment to prevent shallow copies
    Cos(const Cos&) = delete;
    Cos& operator=(const Cos&) = delete;
};

// Class asin (asin(x))
class Asin : public Function {
private:
    Function* argument;

public:
    explicit Asin(Function* argument);

    double evaluate() const override;

    Function* derivative(Variable* var) const override;

    Function* clone() const override;

    // Disable copy constructor and copy assignment to prevent shallow copies
    Asin(const Asin&) = delete;
    Asin& operator=(const Asin&) = delete;
};

// Class acos (acos(x))
class Acos : public Function {
private:
    Function* argument;

public:
    explicit Acos(Function* argument);

    double evaluate() const override;

    Function* derivative(Variable* var) const override;

    Function* clone() const override;

    // Disable copy constructor and copy assignment to prevent shallow copies
    Acos(const Acos&) = delete;
    Acos& operator=(const Acos&) = delete;
};

// Class Tan (Tan(x))
class Tan : public Function {
private:
    Function* argument;

public:
    explicit Tan(Function* argument);

    double evaluate() const override;

    Function* derivative(Variable* var) const override;

    Function* clone() const override;

    // Disable copy constructor and copy assignment to prevent shallow copies
    Tan(const Tan&) = delete;
    Tan& operator=(const Tan&) = delete;
};

// Class Atan (Atan(x))
class Atan : public Function {
private:
    Function* argument;

public:
    explicit Atan(Function* argument);

    double evaluate() const override;

    Function* derivative(Variable* var) const override;

    Function* clone() const override;

    // Disable copy constructor and copy assignment to prevent shallow copies
    Atan(const Atan&) = delete;
    Atan& operator=(const Atan&) = delete;
};

#endif // MINIMIZEROPTIMIZER_FUNCTION_H