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
    double den = denominator->evaluate();
    if (den == 0.0) {
        throw std::runtime_error("Division by zero");
    }
    return numerator->evaluate() / den;
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

// -------------------- Negation Implementations --------------------

Negation::Negation(Function* argument)
        : argument(argument) {}

double Negation::evaluate() const {
    return -1 * argument->evaluate();
}

Function* Negation::derivative(Variable* var) const {
    Function* f_prime = argument->derivative(var);
    return new Multiplication(new Constant(-1), f_prime);
}

Function* Negation::clone() const {
    return new Negation(argument->clone());
}

// -------------------- Abs Implementations --------------------

Abs::Abs(Function* argument)
        : argument(argument) {}

double Abs::evaluate() const {
    return std::abs(argument->evaluate());
}

Function* Abs::derivative(Variable* var) const {
    // d/dx |f(x)| = f'(x) * sign(f(x))
    Function* f_prime = argument->derivative(var);
    Function* sign_fx = new Sign(argument->clone());
    Function* derivative = new Multiplication(f_prime, sign_fx);
    return derivative;
}

Function* Abs::clone() const {
    return new Abs(argument->clone());
}

// -------------------- Sign Implementations --------------------

Sign::Sign(Function* argument)
        : argument(argument) {}

double Sign::evaluate() const {
    double arg_value = argument->evaluate();
    if (arg_value > 0.0) {
        return 1.0;
    } else if (arg_value < 0.0) {
        return -1.0;
    } else {
        return 0.0; // By convention, sign(0) = 0
    }
}

Function* Sign::derivative(Variable* var) const {
    return new Constant(0.0);
}

Function* Sign::clone() const {
    return new Sign(argument->clone());
}


// -------------------- Modulo Implementations --------------------

Mod::Mod(Function* numerator, Function* denominator)
        : numerator(numerator), denominator(denominator) {}

double Mod::evaluate() const {
    double num = numerator->evaluate();
    double den = denominator->evaluate();

    if (den == 0.0) {
        throw std::domain_error("Division by zero in Modulo function.");
    }

    return std::fmod(num, den);
}

Function* Mod::derivative(Variable* var) const {
    return new Constant(0.0);
}

Function* Mod::clone() const {
    return new Mod(numerator->clone(), denominator->clone());
}

// -------------------- Exponential Implementations --------------------

Exp::Exp(Function* exponent)
        : exponent(exponent) {}

double Exp::evaluate() const {
    return std::exp(exponent->evaluate());
}

Function* Exp::derivative(Variable* var) const {
    // (e^{f(x)})' = f'(x) * e^{f(x)}
    Function* f_prime = exponent->derivative(var);
    return new Multiplication(f_prime, this->clone());
}

Function* Exp::clone() const {
    return new Exp(exponent->clone());
}

// -------------------- Ln Implementations --------------------

Ln::Ln(Function* argument)
        : argument(argument) {}

double Ln::evaluate() const {
    double arg_value = argument->evaluate();
    if (arg_value <= 0.0) {
        throw std::runtime_error("Logarithm of non-positive value");
    }
    return std::log(arg_value);
}

Function* Ln::derivative(Variable* var) const {
    // (ln(f(x)))' = f'(x) / f(x)
    Function* f_prime = argument->derivative(var);
    return new Division(f_prime, argument->clone());
}

Function* Ln::clone() const {
    return new Ln(argument->clone());
}

// -------------------- Log Implementations --------------------

Log::Log(Function* base, Function* argument)
        : base(base), argument(argument) {}


double Log::evaluate() const {
    double base_val = base->evaluate();
    double arg_val = argument->evaluate();
    if (base_val <= 0.0 || base_val == 1.0) {
        throw std::runtime_error("Invalid base for logarithm");
    }
    if (arg_val <= 0.0) {
        throw std::runtime_error("Logarithm of non-positive value");
    }
    return std::log(arg_val) / std::log(base_val);
}

Function* Log::derivative(Variable* var) const {
    // d/dx log_b(f(x)) = f'(x) / (f(x) * ln(b))
    Function* f_prime = argument->derivative(var);
    Function* ln_b = new Ln(base->clone());
    Function* denominator = new Multiplication(argument->clone(), ln_b);
    return new Division(f_prime, denominator);
}

Function* Log::clone() const {
    return new Log(base->clone(), argument->clone());
}

// -------------------- Sqrt Implementations --------------------

Sqrt::Sqrt(Function *argument)
        : argument(argument) {}

double Sqrt::evaluate() const {
    double arg_value = argument->evaluate();
    return std::sqrt(arg_value);
}

Function *Sqrt::derivative(Variable* var) const {
    // (sqrt(f(x)))' = f'(x) / (2 * sqrt(f(x)))

    // f'(x)
    Function* f_prime = argument->derivative(var);

    // 2 * sqrt(f(x))
    Function* two = new Constant(2.0);
    Function* sqrt_f = this->clone(); // sqrt(f(x))
    Function* denominator = new Multiplication(two, sqrt_f);

    // f'(x) / (2 * sqrt(f(x)))
    return new Division(f_prime, denominator);
}

Function *Sqrt::clone() const {
    return new Sqrt(argument->clone());
}


// -------------------- Sin Implementations --------------------

Sin::Sin(Function *argument)
        : argument(argument) {}

double Sin::evaluate() const {
    double arg_value = argument->evaluate();
    return std::sin(arg_value);
}

Function *Sin::derivative(Variable* var) const {
    // (sin(f(x))' = f'(x) * cos(f(x)));
    Function* f_prime = argument->derivative(var);
    return new Multiplication(f_prime, new Cos(argument));
}

Function *Sin::clone() const {
    return new Sin(argument->clone());
}

// -------------------- Cos Implementations --------------------

Cos::Cos(Function *argument)
        : argument(argument) {}

double Cos::evaluate() const {
    double arg_value = argument->evaluate();
    return std::cos(arg_value);
}

Function *Cos::derivative(Variable* var) const {
    // (cos(f(x))' = (-1) * f'(x) * sin(f(x)));
    Function *f_prime = argument->derivative(var);
    Function *sin_func = new Sin(argument->clone());
    Function *mul = new Multiplication(f_prime, sin_func);
    return new Multiplication(new Constant(-1), mul);
}

Function *Cos::clone() const {
    return new Cos(argument->clone());
}

// -------------------- Asin Implementations --------------------

Asin::Asin(Function *argument)
        : argument(argument) {}

double Asin::evaluate() const {
    double arg_value = argument->evaluate();
    return std::asin(arg_value);
}

Function *Asin::derivative(Variable* var) const {
    // (acos(f(x)))' = f'(x) / sqrt(1 - f(x)^2)
    Function* f_prime = argument->derivative(var);

    // 1 - f(x)^2
    Function* f_squared = new Multiplication(argument->clone(), argument->clone());
    Function* one_minus_f_squared = new Subtraction(new Constant(1.0), f_squared);

    // sqrt(1 - f(x)^2)
    Function* denominator = new Sqrt(one_minus_f_squared);

    // f'(x) / sqrt(1 - f(x)^2)
    Function* f = new Division(f_prime, denominator);
    return f;
}

Function *Asin::clone() const {
    return new Asin(argument->clone());
}

// -------------------- Acos Implementations --------------------

Acos::Acos(Function *argument)
        : argument(argument) {}

double Acos::evaluate() const {
    double arg_value = argument->evaluate();
    return std::acos(arg_value);
}

Function *Acos::derivative(Variable* var) const {
    // (acos(f(x)))' = -f'(x) / sqrt(1 - f(x)^2)
    Function* f_prime = argument->derivative(var);

    // 1 - f(x)^2
    Function* f_squared = new Multiplication(argument->clone(), argument->clone());
    Function* one_minus_f_squared = new Subtraction(new Constant(1.0), f_squared);

    // sqrt(1 - f(x)^2)
    Function* denominator = new Sqrt(one_minus_f_squared);
    Function* negative_one = new Constant(-1.0);
    // -f'(x) * (1 / sqrt(1 - f(x)^2))
    Function* f = new Division(new Multiplication(negative_one, f_prime), denominator);
    return f;
}

Function *Acos::clone() const {
    return new Acos(argument->clone());
}

// -------------------- Tan Implementations --------------------

Tan::Tan(Function *argument)
        : argument(argument) {}

double Tan::evaluate() const {
    double arg_value = argument->evaluate();
    return std::tan(arg_value);
}

Function *Tan::derivative(Variable* var) const {
    // (tan(f(x)))' = f'(x) / cos^2(f(x))
    Function* f_prime = argument->derivative(var);

    // cos(f(x))
    Function* cos_fx = new Cos(argument->clone());

    // cos(f(x)) * cos(f(x)) = cos^2(f(x))
    Function* cos_fx_squared = new Multiplication(cos_fx->clone(), cos_fx->clone());

    // f'(x) / cos^2(f(x))
    Function* derivative = new Division(f_prime, cos_fx_squared);

    return derivative;
}

Function *Tan::clone() const {
    return new Tan(argument->clone());
}

// -------------------- Atan Implementations --------------------

Atan::Atan(Function *argument)
        : argument(argument) {}

double Atan::evaluate() const {
    double arg_value = argument->evaluate();
    return std::atan(arg_value);
}

Function *Atan::derivative(Variable* var) const {
    // (atan(f(x)))' = f'(x) / (1 + f(x)^2)

    // f'(x)
    Function* f_prime = argument->derivative(var);

    // f(x) * f(x) = f(x)^2
    Function* f_squared = new Multiplication(argument->clone(), argument->clone());

    // 1 + f(x)^2
    Function* one_plus_f_squared = new Addition(new Constant(1.0), f_squared);

    // f'(x) / (1 + f(x)^2)
    Function* derivative = new Division(f_prime, one_plus_f_squared);

    return derivative;
}

Function *Atan::clone() const {
    return new Atan(argument->clone());
}

// -------------------- Max Implementations --------------------

Max::Max(Function *left, Function *right)
: left(left), right(right) {}

double Max::evaluate() const {
    double left_val = left->evaluate();
    double right_val = right->evaluate();
    return std::max(left_val, right_val);
}

Function *Max::derivative(Variable *var) const {
    double left_val = left->evaluate();
    double right_val = right->evaluate();

    if (left_val > right_val) {
        return left->derivative(var);
    } else if (right_val > left_val) {
        return right->derivative(var);
    } else {
        // At points where left == right, derivative is undefined.
        // Here we choose to return a zero function by convention.
        return new Constant(0.0);
    }
}

Function *Max::clone() const {
    return new Max(left->clone(), right->clone());
}

// -------------------- Max Implementations --------------------

Min::Min(Function *left, Function *right)
        : left(left), right(right) {}

double Min::evaluate() const {
    double left_val = left->evaluate();
    double right_val = right->evaluate();
    return std::min(left_val, right_val);
}

Function *Min::derivative(Variable *var) const {
    double left_val = left->evaluate();
    double right_val = right->evaluate();

    if (left_val < right_val) {
        return left->derivative(var);
    } else if (right_val < left_val) {
        return right->derivative(var);
    } else {
        // At points where left == right, derivative is undefined.
        // Here we choose to return a zero function by convention.
        return new Constant(0.0);
    }
}

Function *Min::clone() const {
    return new Min(left->clone(), right->clone());
}
