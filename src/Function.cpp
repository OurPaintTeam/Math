#include "Function.h"

// -------------------- Constant Implementations --------------------

Constant::Constant(double value) : value(value) {}

double Constant::evaluate() const {
    return value;
}

Function* Constant::derivative(Variable* /*var*/) const {
    return new Constant(0.0);
}

Function* Constant::clone() const {
    return new Constant(value);
}

// -------------------- Variable Implementations --------------------

Variable::Variable(double* value) : value(value) {}

double Variable::evaluate() const {
    return *value;
}

Function* Variable::derivative(Variable* var) const {
    if (*this == var) {
        return new Constant(1.0);
    }
    return new Constant(0.0);
}

Function* Variable::clone() const {
    return new Variable(value);
}

bool Variable::operator==(Variable* other) const {
    return this->value == other->value;
}

void Variable::setValue(double value) {
    *this->value = value;
}

// -------------------- Addition Implementations --------------------

Addition::Addition(Function* left, Function* right)
        : left(left), right(right) {}

double Addition::evaluate() const {
    return left->evaluate() + right->evaluate();
}

Function* Addition::derivative(Variable* var) const {
    return new Addition(
            left->derivative(var),
            right->derivative(var)
    );
}

Function* Addition::clone() const {
    return new Addition(left->clone(), right->clone());
}

// -------------------- Subtraction Implementations --------------------

Subtraction::Subtraction(Function* left, Function* right)
        : left(left), right(right) {}



double Subtraction::evaluate() const {
    return left->evaluate() - right->evaluate();
}

Function* Subtraction::derivative(Variable* var) const {
    return new Subtraction(
            left->derivative(var),
            right->derivative(var)
    );
}

Function* Subtraction::clone() const {
    return new Subtraction(left->clone(), right->clone());
}

// -------------------- Multiplication Implementations --------------------

Multiplication::Multiplication(Function* left, Function* right)
        : left(left), right(right) {}


double Multiplication::evaluate() const {
    return left->evaluate() * right->evaluate();
}

Function* Multiplication::derivative(Variable* var) const {
    // Product rule: (f * g)' = f' * g + f * g'
    return new Addition(
            new Multiplication(left->derivative(var), right->clone()),
            new Multiplication(left->clone(), right->derivative(var))
    );
}

Function* Multiplication::clone() const {
    return new Multiplication(left->clone(), right->clone());
}

// -------------------- Division Implementations --------------------

Division::Division(Function* numerator, Function* denominator)
        : numerator(numerator), denominator(denominator) {}



double Division::evaluate() const {
    double denom = denominator->evaluate();
    if (denom == 0.0) {
        throw std::runtime_error("Division by zero");
    }
    return numerator->evaluate() / denom;
}

Function* Division::derivative(Variable* var) const {
    // Quotient rule: (f / g)' = (f' * g - f * g') / g^2
    Function* f_prime = numerator->derivative(var);
    Function* g_prime = denominator->derivative(var);

    Function* numerator_derivative = new Subtraction(
            new Multiplication(f_prime, denominator->clone()),
            new Multiplication(numerator->clone(), g_prime)
    );

    Function* denominator_squared = new Power(
            denominator->clone(),
            new Constant(2.0)
    );

    return new Division(
            numerator_derivative,
            denominator_squared
    );
}

Function* Division::clone() const {
    return new Division(numerator->clone(), denominator->clone());
}

// -------------------- Power Implementations --------------------

Power::Power(Function* base, Function* exponent)
        : base(base), exponent(exponent) {}



double Power::evaluate() const {
    return std::pow(base->evaluate(), exponent->evaluate());
}

Function* Power::derivative(Variable* var) const {
    Function* base_derivative = base->derivative(var);
    Function* exp = new Constant(exponent->evaluate());
    Function* new_power = new Power(base, new Constant(exponent->evaluate() - 1));
    // Собираем производную по правилу цепочки: n * u^(n-1) * u'
    Function* result = new Multiplication(exp, new Multiplication(new_power, base_derivative));
    return result;
}

Function* Power::clone() const {
    return new Power(base->clone(), exponent->clone());
}

// -------------------- Exponential Implementations --------------------

Exponential::Exponential(Function* exponent)
        : exponent(exponent) {}



double Exponential::evaluate() const {
    return std::exp(exponent->evaluate());
}

Function* Exponential::derivative(Variable* var) const {
    // (e^{f(x)})' = f'(x) * e^{f(x)}
    Function* f_prime = exponent->derivative(var);
    return new Multiplication(f_prime, this->clone());
}

Function* Exponential::clone() const {
    return new Exponential(exponent->clone());
}

// -------------------- Logarithm Implementations --------------------

Logarithm::Logarithm(Function* argument)
        : argument(argument) {}


double Logarithm::evaluate() const {
    double arg_value = argument->evaluate();
    if (arg_value <= 0.0) {
        throw std::runtime_error("Logarithm of non-positive value");
    }
    return std::log(arg_value);
}

Function* Logarithm::derivative(Variable* var) const {
    // (ln(f(x)))' = f'(x) / f(x)
    Function* f_prime = argument->derivative(var);
    return new Division(f_prime, argument->clone());
}

Function* Logarithm::clone() const {
    return new Logarithm(argument->clone());
}