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
    if (value == var->value) {
        return new Constant(1.0);
    }
    return new Constant(0.0);
}

Variable* Variable::clone() const{
    return const_cast<Variable*>(this);
}

bool Variable::operator==(Variable* other) const {
    return this->value == other->value;
}

void Variable::setValue(double val) {
    *this->value = val;
}

// -------------------- Addition Implementations --------------------

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

double Division::evaluate() const {
    double den = right->evaluate();
    if (den == 0.0) {
        throw std::runtime_error("Division by zero");
    }
    return left->evaluate() / den;
}

Function* Division::derivative(Variable* var) const {
    // Quotient rule: (f / g)' = (f' * g - f * g') / g^2
    Function* f_prime = left->derivative(var);
    Function* g_prime = right->derivative(var);

    Function* left_derivative = new Subtraction(
            new Multiplication(f_prime, right->clone()),
            new Multiplication(left->clone(), g_prime)
    );

    Function* right_squared = new Power(
            right->clone(),
            new Constant(2.0)
    );

    return new Division(
            left_derivative,
            right_squared
    );
}

Function* Division::clone() const {
    return new Division(left->clone(), right->clone());
}

// -------------------- Power Implementations --------------------

double Power::evaluate() const {
    return std::pow(left->evaluate(), right->evaluate());
}

Function* Power::derivative(Variable* var) const {
    Function* left_derivative = left->derivative(var);
    Function* exp = new Constant(right->evaluate());
    Function* new_power = new Power(left->clone(), new Constant(right->evaluate() - 1));
    // n * u^(n-1) * u'
    return new Multiplication(exp, new Multiplication(new_power, left_derivative));
}

Function* Power::clone() const {
    return new Power(left->clone(), right->clone());
}

// -------------------- Negation Implementations --------------------

double Negation::evaluate() const {
    return -1 * operand->evaluate();
}

Function* Negation::derivative(Variable* var) const {
    Function* f_prime = operand->derivative(var);
    return new Multiplication(new Constant(-1), f_prime);
}

Function* Negation::clone() const {
    return new Negation(operand->clone());
}

// -------------------- Abs Implementations --------------------

double Abs::evaluate() const {
    return std::abs(operand->evaluate());
}

Function* Abs::derivative(Variable* var) const {
    // d/dx |f(x)| = f'(x) * sign(f(x))
    Function* f_prime = operand->derivative(var);
    Function* sign_fx = new Sign(operand->clone());
    Function* derivative = new Multiplication(f_prime, sign_fx);
    return derivative;
}

Function* Abs::clone() const {
    return new Abs(operand->clone());
}

// -------------------- Sign Implementations --------------------

double Sign::evaluate() const {
    double arg_value = operand->evaluate();
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
    return new Sign(operand->clone());
}


// -------------------- Modulo Implementations --------------------

double Mod::evaluate() const {
    double num = left->evaluate();
    double den = right->evaluate();

    if (den == 0.0) {
        throw std::domain_error("Division by zero in Modulo function.");
    }

    return std::fmod(num, den);
}

Function* Mod::derivative(Variable* var) const {
    return new Constant(0.0);
}

Function* Mod::clone() const {
    return new Mod(left->clone(), right->clone());
}

// -------------------- rightial Implementations --------------------

double Exp::evaluate() const {
    return std::exp(operand->evaluate());
}

Function* Exp::derivative(Variable* var) const {
    // (e^{f(x)})' = f'(x) * e^{f(x)}
    Function* f_prime = operand->derivative(var);
    return new Multiplication(f_prime, this->clone());
}

Function* Exp::clone() const {
    return new Exp(operand->clone());
}

// -------------------- Ln Implementations --------------------

double Ln::evaluate() const {
    double arg_value = operand->evaluate();
    if (arg_value <= 0.0) {
        throw std::runtime_error("Logarithm of non-positive value");
    }
    return std::log(arg_value);
}

Function* Ln::derivative(Variable* var) const {
    // (ln(f(x)))' = f'(x) / f(x)
    Function* f_prime = operand->derivative(var);
    return new Division(f_prime, operand->clone());
}

Function* Ln::clone() const {
    return new Ln(operand->clone());
}

// -------------------- Log Implementations --------------------


double Log::evaluate() const {
    double left_val = left->evaluate();
    double arg_val = right->evaluate();
    if (left_val <= 0.0 || left_val == 1.0) {
        throw std::runtime_error("Invalid left for logarithm");
    }
    if (arg_val <= 0.0) {
        throw std::runtime_error("Logarithm of non-positive value");
    }
    return std::log(arg_val) / std::log(left_val);
}

Function* Log::derivative(Variable* var) const {
    // d/dx log_b(f(x)) = f'(x) / (f(x) * ln(b))
    Function* f_prime = right->derivative(var);
    Function * right_clone = right->clone();
    Function* ln_b = new Ln(left->clone());
    Function* right = new Multiplication(right_clone, ln_b);
    return new Division(f_prime, right);
}

Function* Log::clone() const {
    return new Log(left->clone(), right->clone());
}

// -------------------- Sqrt Implementations --------------------

double Sqrt::evaluate() const {
    double arg_value = operand->evaluate();
    return std::sqrt(arg_value);
}

Function *Sqrt::derivative(Variable* var) const {
    // (sqrt(f(x)))' = f'(x) / (2 * sqrt(f(x)))

    // f'(x)
    Function* f_prime = operand->derivative(var);

    // 2 * sqrt(f(x))
    Function* two = new Constant(2.0);
    Function* sqrt_f = this->clone(); // sqrt(f(x))
    Function* right = new Multiplication(two, sqrt_f);

    // f'(x) / (2 * sqrt(f(x)))
    return new Division(f_prime, right);
}

Function *Sqrt::clone() const {
    return new Sqrt(operand->clone());
}


// -------------------- Sin Implementations --------------------

double Sin::evaluate() const {
    double arg_value = operand->evaluate();
    return std::sin(arg_value);
}

Function *Sin::derivative(Variable* var) const {
    // (sin(f(x))' = f'(x) * cos(f(x)));
    Function* f_prime = operand->derivative(var);
    return new Multiplication(f_prime, new Cos(operand->clone()));
}

Function *Sin::clone() const {
    return new Sin(operand->clone());
}

// -------------------- Cos Implementations --------------------

double Cos::evaluate() const {
    double arg_value = operand->evaluate();
    return std::cos(arg_value);
}

Function *Cos::derivative(Variable* var) const {
    // (cos(f(x))' = (-1) * f'(x) * sin(f(x)));
    Function *f_prime = operand->derivative(var);
    Function *sin_func = new Sin(operand->clone());
    Function *mul = new Multiplication(f_prime, sin_func);
    return new Multiplication(new Constant(-1), mul);
}

Function *Cos::clone() const {
    return new Cos(operand->clone());
}

// -------------------- Asin Implementations --------------------


double Asin::evaluate() const {
    double arg_value = operand->evaluate();
    return std::asin(arg_value);
}

Function *Asin::derivative(Variable* var) const {
    // (acos(f(x)))' = f'(x) / sqrt(1 - f(x)^2)
    Function* f_prime = operand->derivative(var);

    // 1 - f(x)^2
    Function* f_squared = new Multiplication(operand->clone(), operand->clone());
    Function* one_minus_f_squared = new Subtraction(new Constant(1.0), f_squared);

    // sqrt(1 - f(x)^2)
    Function* right = new Sqrt(one_minus_f_squared);

    // f'(x) / sqrt(1 - f(x)^2)
    Function* f = new Division(f_prime, right);
    return f;
}

Function *Asin::clone() const {
    return new Asin(operand->clone());
}

// -------------------- Acos Implementations --------------------

double Acos::evaluate() const {
    double arg_value = operand->evaluate();
    return std::acos(arg_value);
}

Function *Acos::derivative(Variable* var) const {
    // (acos(f(x)))' = -f'(x) / sqrt(1 - f(x)^2)
    Function* f_prime = operand->derivative(var);

    // 1 - f(x)^2
    Function* f_squared = new Multiplication(operand->clone(), operand->clone());
    Function* one_minus_f_squared = new Subtraction(new Constant(1.0), f_squared);

    // sqrt(1 - f(x)^2)
    Function* right = new Sqrt(one_minus_f_squared);
    Function* negative_one = new Constant(-1.0);
    // -f'(x) * (1 / sqrt(1 - f(x)^2))
    Function* f = new Division(new Multiplication(negative_one, f_prime), right);
    return f;
}

Function *Acos::clone() const {
    return new Acos(operand->clone());
}

// -------------------- Tan Implementations --------------------


double Tan::evaluate() const {
    double arg_value = operand->evaluate();
    return std::tan(arg_value);
}

Function *Tan::derivative(Variable* var) const {
    // (tan(f(x)))' = f'(x) / cos^2(f(x))
    Function* f_prime = operand->derivative(var);

    // cos(f(x))
    Function* cos_fx = new Cos(operand->clone());

    // cos(f(x)) * cos(f(x)) = cos^2(f(x))
    Function* cos_fx_squared = new Multiplication(cos_fx, cos_fx->clone());

    // f'(x) / cos^2(f(x))
    Function* derivative = new Division(f_prime, cos_fx_squared);

    return derivative;
}

Function *Tan::clone() const {
    return new Tan(operand->clone());
}

// -------------------- Atan Implementations --------------------

double Atan::evaluate() const {
    double arg_value = operand->evaluate();
    return std::atan(arg_value);
}

Function *Atan::derivative(Variable* var) const {
    // (atan(f(x)))' = f'(x) / (1 + f(x)^2)

    // f'(x)
    Function* f_prime = operand->derivative(var);

    // f(x) * f(x) = f(x)^2
    Function* f_squared = new Multiplication(operand->clone(), operand->clone());

    // 1 + f(x)^2
    Function* one_plus_f_squared = new Addition(new Constant(1.0), f_squared);

    // f'(x) / (1 + f(x)^2)
    Function* derivative = new Division(f_prime, one_plus_f_squared);

    return derivative;
}

Function *Atan::clone() const {
    return new Atan(operand->clone());
}

// -------------------- Cot Implementations --------------------

double Cot::evaluate() const {
    double arg_value = operand->evaluate();
    return 1.0 / std::tan(arg_value);
}

Function *Cot::derivative(Variable* var) const {
    // (cot(f(x)))' = -f'(x) / sin^2(f(x))

    // f'(x)
    Function* f_prime = operand->derivative(var);

    // sin(f(x))
    Function* sin_fx = new Sin(operand->clone());

    // sin(f(x)) * sin(f(x)) = sin^2(f(x))
    Function* sin_fx_squared = new Multiplication(sin_fx, sin_fx->clone());

    // -f'(x)
    Function* negative_f_prime = new Negation(f_prime);

    // -f'(x) / sin^2(f(x))
    Function* derivative = new Division(negative_f_prime, sin_fx_squared);

    return derivative;
}

Function *Cot::clone() const {
    return new Cot(operand->clone());
}

// -------------------- Acot Implementations --------------------
double Acot::evaluate() const {
    double arg_value = operand->evaluate();
    // Acot(x) = π/2 - atan(x)
    return (M_PI / 2.0) - std::atan(arg_value);
}

Function *Acot::derivative(Variable* var) const {
    // (acot(f(x)))' = -f'(x) / (1 + f(x)^2)

    // f'(x)
    Function* f_prime = operand->derivative(var);

    // f(x) * f(x) = f(x)^2
    Function* f_squared = new Multiplication(operand->clone(), operand->clone());

    // 1 + f(x)^2
    Function* one_plus_f_squared = new Addition(new Constant(1.0), f_squared);

    // -f'(x)
    Function* negative_f_prime = new Negation(f_prime);

    // -f'(x) / (1 + f(x)^2)
    Function* derivative = new Division(negative_f_prime, one_plus_f_squared);

    return derivative;
}

Function *Acot::clone() const {
    return new Acot(operand->clone());
}

// -------------------- Max Implementations --------------------


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
