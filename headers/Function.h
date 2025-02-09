#ifndef MINIMIZEROPTIMIZER_HEADERS_FUNCTION_H_
#define MINIMIZEROPTIMIZER_HEADERS_FUNCTION_H_
#include <cmath>
#include <stdexcept>

// Function

//  Unary
//  Binary
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
//  Max
//  Min





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
// Class for unary operation
class Unary: public Function {
protected:
    Function* operand;
public:
    Unary(Function* op): operand(op){}
    ~Unary(){
        delete operand;
    }
    virtual double evaluate() const override = 0;
    virtual Function* derivative(Variable* var) const override = 0;
    virtual Function* clone() const = 0;
};
// Class for binary operation
class Binary: public Function {
protected:
    Function* left;
    Function* right;
public:
    Binary(Function* l, Function* r): left(l), right(r) {}
    ~Binary(){
        delete left;
        delete right;
    }
    virtual double evaluate() const override = 0;
    virtual Function* derivative(Variable* var) const override = 0;
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
class Addition : public Binary {

public:
    Addition(Function* left, Function* right) : Binary(left, right){}

    double evaluate() const override;

    Function* derivative(Variable* var) const override;

    Function* clone() const override;


    // Disable copy constructor and copy assignment to prevent shallow copies
    Addition(const Addition&) = delete;
    Addition& operator=(const Addition&) = delete;
};

// Class Subtraction
class Subtraction : public Binary {

public:
    Subtraction(Function* left, Function* right): Binary(left, right) {}

    double evaluate() const override;

    Function* derivative(Variable* var) const override;

    Function* clone() const override;



    // Disable copy constructor and copy assignment to prevent shallow copies
    Subtraction(const Subtraction&) = delete;
    Subtraction& operator=(const Subtraction&) = delete;
};

// Class Multiplication
class Multiplication : public Binary {
public:
    Multiplication(Function* left, Function* right): Binary(left, right){}

    double evaluate() const override;

    Function* derivative(Variable* var) const override;

    Function* clone() const override;


    // Disable copy constructor and copy assignment to prevent shallow copies
    Multiplication(const Multiplication&) = delete;
    Multiplication& operator=(const Multiplication&) = delete;
};

// Class Division
class Division : public Binary {
public:
    Division(Function* numerator, Function* denominator): Binary(numerator, denominator){}

    double evaluate() const override;

    Function* derivative(Variable* var) const override;

    Function* clone() const override;



    // Disable copy constructor and copy assignment to prevent shallow copies
    Division(const Division&) = delete;
    Division& operator=(const Division&) = delete;
};

// Class Power
class Power : public Binary {
public:
    Power(Function* base, Function* exponent): Binary(base, exponent){}

    double evaluate() const override;

    Function* derivative(Variable* var) const override;

    Function* clone() const override;



    // Disable copy constructor and copy assignment to prevent shallow copies
    Power(const Power&) = delete;
    Power& operator=(const Power&) = delete;
};

// Class Negation
class Negation : public Unary {
public:
    explicit Negation(Function* argument): Unary(argument){}

    double evaluate() const override;

    Function* derivative(Variable* var) const override;

    Function* clone() const override;



    // Disable copy constructor and copy assignment to prevent shallow copies
    Negation(const Negation&) = delete;
    Negation& operator=(const Negation&) = delete;
};

// Class Abs
class Abs : public Unary {
public:
    explicit Abs(Function* argument): Unary(argument){}

    double evaluate() const override;

    Function* derivative(Variable* var) const override;

    Function* clone() const override;



    // Disable copy constructor and copy assignment to prevent shallow copies
    Abs(const Abs&) = delete;
    Abs& operator=(const Abs&) = delete;
};

// Class Sign
class Sign : public Unary {
public:
    explicit Sign(Function* argument):Unary(argument){}

    double evaluate() const override;

    Function* derivative(Variable* var) const override;

    Function* clone() const override;



    // Disable copy constructor and copy assignment to prevent shallow copies
    Sign(const Sign&) = delete;
    Sign& operator=(const Sign&) = delete;
};

// Class Modulo
class Mod : public Binary {
private:
    Function* numerator;
    Function* denominator;

public:
    explicit Mod(Function* numerator, Function* denominator): Binary(numerator, denominator){}

    double evaluate() const override;

    Function* derivative(Variable* var) const override;

    Function* clone() const override;



    // Disable copy constructor and copy assignment to prevent shallow copies
    Mod(const Mod&) = delete;
    Mod& operator=(const Mod&) = delete;
};

// Class Exponential (e^x)
class Exp : public Unary {
public:
    explicit Exp(Function* exponent):Unary(exponent){}

    double evaluate() const override;

    Function* derivative(Variable* var) const override;

    Function* clone() const override;



    // Disable copy constructor and copy assignment to prevent shallow copies
    Exp(const Exp&) = delete;
    Exp& operator=(const Exp&) = delete;
};

// Class Ln (ln(x))
class Ln : public Unary {
public:
    explicit Ln(Function* argument): Unary(argument){}

    double evaluate() const override;

    Function* derivative(Variable* var) const override;

    Function* clone() const override;



    // Disable copy constructor and copy assignment to prevent shallow copies
    Ln(const Ln&) = delete;
    Ln& operator=(const Ln&) = delete;
};

// Class Log (log(base, x))
class Log : public Binary {

public:
    explicit Log(Function* base, Function* argument): Binary(base, argument){}

    double evaluate() const override;

    Function* derivative(Variable* var) const override;

    Function* clone() const override;



    // Disable copy constructor and copy assignment to prevent shallow copies
    Log(const Ln&) = delete;
    Log& operator=(const Log&) = delete;
};

// class sqrt (sqrt(x))
class Sqrt : public Unary {
public:
    explicit Sqrt(Function* argument):Unary(argument){}

    double evaluate() const override;

    Function* derivative(Variable* var) const override;

    Function* clone() const override;



    // Disable copy constructor and copy assignment to prevent shallow copies
    Sqrt(const Sqrt&) = delete;
    Sqrt& operator=(const Sqrt&) = delete;
};

// Class sin (sin(x))
class Sin : public Unary {
public:
    explicit Sin(Function* argument): Unary(argument){}

    double evaluate() const override;

    Function* derivative(Variable* var) const override;

    Function* clone() const override;

    // Disable copy constructor and copy assignment to prevent shallow copies
    Sin(const Sin&) = delete;
    Sin& operator=(const Sin&) = delete;
};

// Class cos (cos(x))
class Cos : public Unary {
public:
    explicit Cos(Function* argument):Unary(argument){}

    double evaluate() const override;

    Function* derivative(Variable* var) const override;

    Function* clone() const override;

    // Disable copy constructor and copy assignment to prevent shallow copies
    Cos(const Cos&) = delete;
    Cos& operator=(const Cos&) = delete;
};

// Class asin (asin(x))
class Asin : public Unary {
public:
    explicit Asin(Function* argument): Unary(argument){}

    double evaluate() const override;

    Function* derivative(Variable* var) const override;

    Function* clone() const override;

    // Disable copy constructor and copy assignment to prevent shallow copies
    Asin(const Asin&) = delete;
    Asin& operator=(const Asin&) = delete;
};

// Class acos (acos(x))
class Acos : public Unary {
public:
    explicit Acos(Function* argument): Unary(argument){}

    double evaluate() const override;

    Function* derivative(Variable* var) const override;

    Function* clone() const override;

    // Disable copy constructor and copy assignment to prevent shallow copies
    Acos(const Acos&) = delete;
    Acos& operator=(const Acos&) = delete;
};

// Class Tan (Tan(x))
class Tan : public Unary {
public:
    explicit Tan(Function* argument):Unary(argument){}

    double evaluate() const override;

    Function* derivative(Variable* var) const override;

    Function* clone() const override;

    // Disable copy constructor and copy assignment to prevent shallow copies
    Tan(const Tan&) = delete;
    Tan& operator=(const Tan&) = delete;
};

// Class Atan (Atan(x))
class Atan : public Unary {
public:
    explicit Atan(Function* argument):Unary(argument){}

    double evaluate() const override;

    Function* derivative(Variable* var) const override;

    Function* clone() const override;

    // Disable copy constructor and copy assignment to prevent shallow copies
    Atan(const Atan&) = delete;
    Atan& operator=(const Atan&) = delete;
};

// Class Cot (Cot(x))
class Cot : public Unary {
public:
    explicit Cot(Function* argument):Unary(argument){}

    double evaluate() const override;

    Function* derivative(Variable* var) const override;

    Function* clone() const override;

    // Disable copy constructor and copy assignment to prevent shallow copies
    Cot(const Cot&) = delete;
    Cot& operator=(const Cot&) = delete;
};

// Class Acot (Acot(x))
class Acot : public Unary {
public:
    explicit Acot(Function* argument):Unary(argument){}

    double evaluate() const override;

    Function* derivative(Variable* var) const override;

    Function* clone() const override;

    // Disable copy constructor and copy assignment to prevent shallow copies
    Acot(const Acot&) = delete;
    Acot& operator=(const Acot&) = delete;
};

// Class Max (Max(x, y))
class Max : public Binary {

public:
    explicit Max(Function* left, Function* right):Binary(left, right){}

    double evaluate() const override;

    Function* derivative(Variable* var) const override;

    Function* clone() const override;

    // Disable copy constructor and copy assignment to prevent shallow copies
    Max(const Max&) = delete;
    Max& operator=(const Max&) = delete;
};

// Class Min (Min(x, y))
class Min : public Binary {
public:
    explicit Min(Function* left, Function* right):Binary(left, right){}

    double evaluate() const override;

    Function* derivative(Variable* var) const override;

    Function* clone() const override;

    // Disable copy constructor and copy assignment to prevent shallow copies
    Min(const Min&) = delete;
    Min& operator=(const Min&) = delete;
};

#endif // ! MINIMIZEROPTIMIZER_HEADERS_FUNCTION_H_