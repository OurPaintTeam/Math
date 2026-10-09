#include "ErrorFunction.h"
#include <array>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>

namespace {
constexpr std::size_t N = 8;
constexpr double undefined = std::numeric_limits<double>::infinity();

// Second-order forward differentiation of the same expression used for values.
// Seeds are coordinate identities, so repeated arguments sum automatically.
struct Jet {
    long double value = 0;
    std::array<long double, N> g{};
    std::array<long double, N * N> h{};
    Jet(long double v = 0) : value(v) {}
};
long double euclideanNorm(std::span<const long double> coordinates) {
    long double scale = 0;
    for (long double coordinate : coordinates) {
        const long double magnitude = std::abs(coordinate);
        if (!std::isfinite(magnitude)) return magnitude;
        scale = std::max(scale,magnitude);
    }
    if (scale == 0) return 0;
    // Scaling also keeps intermediates finite when long double has double's range.
    long double sumSquares = 0;
    for (long double coordinate : coordinates) {
        const long double scaled = coordinate/scale;
        sumSquares += scaled*scaled;
    }
    return scale*std::sqrt(sumSquares);
}
Jet operator+(const Jet& a, const Jet& b) {
    Jet r(a.value + b.value);
    for (std::size_t i = 0; i < N; ++i) r.g[i] = a.g[i] + b.g[i];
    for (std::size_t i = 0; i < N*N; ++i) r.h[i] = a.h[i] + b.h[i];
    return r;
}
Jet operator-(const Jet& a, const Jet& b) {
    Jet r(a.value - b.value);
    for (std::size_t i = 0; i < N; ++i) r.g[i] = a.g[i] - b.g[i];
    for (std::size_t i = 0; i < N*N; ++i) r.h[i] = a.h[i] - b.h[i];
    return r;
}
Jet operator*(const Jet& a, const Jet& b) {
    Jet r(a.value * b.value);
    for (std::size_t i = 0; i < N; ++i) {
        r.g[i] = a.g[i]*b.value + a.value*b.g[i];
        for (std::size_t j = 0; j < N; ++j)
            r.h[i*N+j] = a.h[i*N+j]*b.value + a.g[i]*b.g[j] +
                         a.g[j]*b.g[i] + a.value*b.h[i*N+j];
    }
    return r;
}
Jet operator/(const Jet& a, const Jet& b) {
    Jet r(a.value / b.value);
    for (std::size_t i = 0; i < N; ++i) r.g[i] = (a.g[i] - r.value*b.g[i])/b.value;
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = 0; j < N; ++j)
            r.h[i*N+j] = (a.h[i*N+j] - r.g[i]*b.g[j] - r.g[j]*b.g[i] -
                         r.value*b.h[i*N+j])/b.value;
    return r;
}
Jet norm(const Jet& x, const Jet& y) {
    const std::array<long double,2> coordinates{x.value,y.value};
    const long double length = euclideanNorm(coordinates);
    if (length == 0) return Jet(0); // Selected zero subgradient at the norm cusp.
    Jet r(length);
    const long double ux = x.value/length, uy = y.value/length;
    for (std::size_t i = 0; i < N; ++i) r.g[i] = ux*x.g[i] + uy*y.g[i];
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = 0; j < N; ++j)
            r.h[i*N+j] = ux*x.h[i*N+j] + uy*y.h[i*N+j] +
                ((uy*x.g[i] - ux*y.g[i])*(uy*x.g[j] - ux*y.g[j]))/length;
    return r;
}

double checked(long double value) {
    if (!std::isfinite(value) || std::abs(value) > std::numeric_limits<double>::max()) return undefined;
    return static_cast<double>(value);
}
} // namespace

struct ErrorFunction::State {
    static std::size_t arity(Equation kind) {
        switch (kind) {
            case Equation::FixCoordinate: case Equation::CircleRadius: return 1;
            case Equation::PointPointDistance: case Equation::PointOnPoint:
            case Equation::Vertical: case Equation::Horizontal: return 4;
            case Equation::PointOnCircle: return 5;
            case Equation::PointLineDistance: case Equation::PointOnLine:
            case Equation::ArcBisector: return 6;
            case Equation::SegmentCircleDistance: case Equation::SegmentOnCircle: return 7;
            case Equation::Parallel: case Equation::Perpendicular: case Equation::Angle:
            case Equation::EqualLength: return 8;
            case Equation::SegmentInCircle:
                throw std::logic_error("SegmentInCircle is unsupported");
        }
        throw std::invalid_argument("Unknown constraint kind");
    }
    static void validateTarget(Equation kind, double target) {
        if (!std::isfinite(target)) throw std::invalid_argument("Constraint target must be finite");
        if (kind == Equation::CircleRadius && target <= 0)
            throw std::invalid_argument("Circle radius target must be positive");
        if ((kind == Equation::PointPointDistance || kind == Equation::SegmentCircleDistance) && target < 0)
            throw std::invalid_argument("Distance/clearance must be non-negative");
        if (kind == Equation::Angle && (target < 0 || target > std::numbers::pi))
            throw std::invalid_argument("Angle must be in [0, pi] radians");
        if (kind != Equation::PointPointDistance && kind != Equation::SegmentCircleDistance &&
            kind != Equation::PointLineDistance && kind != Equation::Angle &&
            kind != Equation::FixCoordinate && kind != Equation::CircleRadius && target != 0)
            throw std::invalid_argument("This equation has no target parameter");
    }
    static bool validRadii(Equation kind, const std::vector<double*>& coordinates) {
        const std::size_t count = (kind == Equation::PointOnCircle || kind == Equation::SegmentOnCircle ||
               kind == Equation::SegmentCircleDistance || kind == Equation::CircleRadius ? 1 : 0);
        for (std::size_t i = coordinates.size() - count; i < coordinates.size(); ++i)
            if (!std::isfinite(*coordinates[i]) || *coordinates[i] <= 0) return false;
        return true;
    }

    static Jet equation(Equation kind, const std::vector<Jet>& x, double target) {
        switch (kind) {
            case Equation::FixCoordinate: return x[0] - Jet(target);
            case Equation::CircleRadius: return x[0] - Jet(target);
            case Equation::EqualLength:
                return norm(x[2]-x[0], x[3]-x[1]) - norm(x[6]-x[4], x[7]-x[5]);
            case Equation::PointPointDistance: case Equation::PointOnPoint:
                return norm(x[2]-x[0], x[3]-x[1]) - Jet(target);
            case Equation::PointOnCircle:
                return norm(x[0]-x[2], x[1]-x[3]) - x[4];
            case Equation::SegmentOnCircle:
                return norm(norm(x[0]-x[4], x[1]-x[5])-x[6],
                            norm(x[2]-x[4], x[3]-x[5])-x[6]);
            case Equation::PointLineDistance: case Equation::PointOnLine: {
                const Jet dx = x[4]-x[2], dy = x[5]-x[3], length = norm(dx,dy);
                if (length.value == 0) return norm(x[0]-x[2],x[1]-x[3])-Jet(target);
                return (x[0]-x[2])*(dy/length) - (x[1]-x[3])*(dx/length) - Jet(target);
            }
            case Equation::SegmentCircleDistance: {
                const Jet dx = x[2]-x[0], dy = x[3]-x[1], length = norm(dx,dy);
                if (length.value == 0) return norm(x[0]-x[4],x[1]-x[5])-x[6]-Jet(target);
                const Jet ux = dx/length, uy = dy/length;
                const Jet projection = (x[4]-x[0])*ux + (x[5]-x[1])*uy;
                if (projection.value <= 0) return norm(x[0]-x[4],x[1]-x[5])-x[6]-Jet(target);
                if (projection.value >= length.value) return norm(x[2]-x[4],x[3]-x[5])-x[6]-Jet(target);
                const Jet t = projection/length;
                return norm((x[0]-x[4])+t*dx,(x[1]-x[5])+t*dy)-x[6]-Jet(target);
            }
            case Equation::Vertical: case Equation::Horizontal: {
                const Jet dx = x[2]-x[0], dy = x[3]-x[1], length = norm(dx,dy);
                if (length.value == 0) return Jet(undefined);
                return (kind == Equation::Vertical ? dx : dy)/length;
            }
            case Equation::Parallel: case Equation::Perpendicular: case Equation::Angle: {
                const Jet dx = x[2]-x[0], dy = x[3]-x[1], length = norm(dx,dy);
                const Jet ex = x[6]-x[4], ey = x[7]-x[5], otherLength = norm(ex,ey);
                if (length.value == 0 || otherLength.value == 0) return Jet(undefined);
                const Jet ux = dx/length, uy = dy/length, vx = ex/otherLength, vy = ey/otherLength;
                if (kind == Equation::Parallel) return ux*vy-uy*vx;
                return ux*vx+uy*vy - Jet(kind == Equation::Angle ? std::cos(target) : 0);
            }
            case Equation::ArcBisector: {
                const Jet dx = x[2]-x[0], dy = x[3]-x[1], length = norm(dx,dy);
                if (length.value == 0) return Jet(undefined);
                // Differences before averaging avoid overflowing a midpoint sum.
                const Jet mx = (x[4]-x[0])*Jet(0.5)+(x[4]-x[2])*Jet(0.5);
                const Jet my = (x[5]-x[1])*Jet(0.5)+(x[5]-x[3])*Jet(0.5);
                return (dx/length)*mx+(dy/length)*my;
            }
            case Equation::SegmentInCircle: break;
        }
        throw std::logic_error("Unsupported constraint equation");
    }

    Equation kind;
    std::vector<double*> variables, unique;
    double target, weight = 1;
    std::size_t revision = 0;
    mutable std::size_t invalidEvaluations = 0;
    mutable std::vector<double> cachedValues;
    mutable Jet result;
    mutable bool dirty = true;

    const Jet& current() const {
        bool changed = dirty || cachedValues.size() != variables.size();
        for (std::size_t i = 0; !changed && i < variables.size(); ++i)
            changed = *variables[i] != cachedValues[i];
        if (!changed) return result;
        std::vector<Jet> x;
        cachedValues.clear();
        bool valid = true;
        for (double* coordinate : variables) {
            cachedValues.push_back(*coordinate);
            valid = valid && std::isfinite(*coordinate);
            Jet seed(*coordinate);
            seed.g[std::find(unique.begin(),unique.end(),coordinate)-unique.begin()] = 1;
            x.push_back(seed);
        }
        if (!validRadii(kind,variables)) valid = false;
        result = valid ? equation(kind,x,target) : Jet(undefined);
        if (!std::isfinite(checked(result.value))) { result = Jet(undefined); ++invalidEvaluations; }
        dirty = false;
        return result;
    }
    std::size_t index(double* variable) const {
        return std::find(unique.begin(),unique.end(),variable)-unique.begin();
    }
};

ErrorFunction::ErrorFunction(Equation kind, std::vector<double*> variables, double target)
    : _state(std::make_shared<State>()) {
    if (variables.size() != State::arity(kind)) throw std::invalid_argument("Wrong constraint variable count");
    State::validateTarget(kind,target);
    for (double* variable : variables) {
        if (!variable || !std::isfinite(*variable)) throw std::invalid_argument("Constraint coordinates must be finite and non-null");
        if (std::find(_state->unique.begin(),_state->unique.end(),variable) == _state->unique.end())
            _state->unique.push_back(variable);
    }
    if (!State::validRadii(kind,variables)) throw std::invalid_argument("Circle radii must be positive");
    _state->kind = kind;
    _state->variables = std::move(variables);
    _state->target = target;
    for (double* coordinate : _state->variables) _variables.emplace_back(coordinate);
}
double ErrorFunction::evaluate() const {
    if (_weighted && weight() == 0) return 0;
    if (_withRespectTo.empty()) return _weighted ? weightedValue() : checked(_state->current().value);
    const auto i = _state->index(_withRespectTo[0]);
    if (i == _state->unique.size()) return 0;
    const auto& result = _state->current();
    long double value = result.g[i];
    if (_withRespectTo.size() == 2) {
        const auto j = _state->index(_withRespectTo[1]);
        if (j == _state->unique.size()) return 0;
        value = result.h[i*N+j];
    }
    const double converted = checked(value * (_weighted ? weight() : 1));
    if (!std::isfinite(converted)) ++_state->invalidEvaluations;
    return converted;
}
ErrorFunction::Gradient ErrorFunction::gradient() const {
    Gradient result;
    const auto& jet = _state->current();
    for (std::size_t i = 0; i < _state->unique.size(); ++i) {
        const double value = checked(jet.g[i]);
        if (!std::isfinite(value)) ++_state->invalidEvaluations;
        result[_state->unique[i]] = value;
    }
    return result;
}
double ErrorFunction::secondDerivative(double* first, double* second) const {
    const auto i = _state->index(first), j = _state->index(second);
    if (i == _state->unique.size() || j == _state->unique.size()) return 0;
    const double value = checked(_state->current().h[i*N+j]);
    if (!std::isfinite(value)) ++_state->invalidEvaluations;
    return value;
}
::Function* ErrorFunction::derivative(Variable* variable) const {
    if (!variable || !variable->value) throw std::invalid_argument("Null derivative variable");
    if (_state->index(variable->value) == _state->unique.size()) return new Constant(0);
    if (_withRespectTo.size() == 2) throw std::logic_error("ErrorFunction derivatives above order two are unsupported");
    auto derivative = std::make_unique<ErrorFunction>(*this);
    derivative->_withRespectTo.push_back(variable->value);
    return derivative.release();
}
ErrorFunction* ErrorFunction::clone() const { return new ErrorFunction(*this); }
std::string ErrorFunction::to_string() const { return "constraint("+std::to_string(static_cast<int>(_state->kind))+")"; }
const std::vector<double*>& ErrorFunction::variables() const { return _state->variables; }
bool ErrorFunction::assignment(double*& variable, double& target) const {
    if (_state->kind != Equation::FixCoordinate) return false;
    variable = variables()[0]; target = _state->target; return true;
}
void ErrorFunction::setTarget(double target) { State::validateTarget(_state->kind,target); _state->target = target; _state->dirty = true; ++_state->revision; }
double ErrorFunction::weight() const { return _state->weight; }
std::size_t ErrorFunction::revision() const { return _state->revision; }
std::size_t ErrorFunction::invalidEvaluations() const { return _state->invalidEvaluations; }
void ErrorFunction::setWeight(double weight) {
    if (!std::isfinite(weight) || weight < 0) throw std::invalid_argument("Weight must be finite and non-negative");
    _state->weight = weight;
    ++_state->revision;
}
double ErrorFunction::weightedValue() const {
    if (weight() == 0) return 0;
    const double result = checked(_state->current().value*static_cast<long double>(weight()));
    if (!std::isfinite(result)) ++_state->invalidEvaluations;
    return result;
}
ErrorFunction::Gradient ErrorFunction::weightedGradient() const {
    if (weight() == 0) { Gradient g; for (double* v : variables()) g[v] = 0; return g; }
    Gradient g;
    const auto& jet = _state->current();
    for (std::size_t i = 0; i < _state->unique.size(); ++i) {
        const double value = checked(jet.g[i]*static_cast<long double>(weight()));
        if (!std::isfinite(value)) ++_state->invalidEvaluations;
        g[_state->unique[i]] = value;
    }
    return g;
}
::Function* ErrorFunction::weightedFunction() const {
    auto function = std::make_unique<ErrorFunction>(*this);
    function->_weighted = true;
    return function.release();
}
bool ErrorFunction::satisfied(double tolerance) const {
    if (!std::isfinite(tolerance) || tolerance < 0) throw std::invalid_argument("Tolerance must be finite and non-negative");
    const double value = weightedValue();
    return std::isfinite(value) && std::abs(value) <= tolerance;
}


std::vector<double*> ErrorFunction::coordinatePointers(const std::vector<Variable*>& variables) {
    std::vector<double*> coordinates;
    for (const auto* variable : variables) {
        if (!variable || !variable->value) throw std::invalid_argument("Null ErrorFunction variable");
        coordinates.push_back(variable->value);
    }
    return coordinates;
}
std::vector<Variable*> ErrorFunction::getVariables() {
    std::vector<Variable*> variables;
    for (auto& variable : _variables) variables.push_back(&variable);
    return variables;
}

double PointPointDistanceError::distance(std::span<const double> first, std::span<const double> second) {
    if (first.size() != second.size()) throw std::invalid_argument("Point dimensions must agree");
    std::vector<long double> differences;
    differences.reserve(first.size());
    for (std::size_t i = 0; i < first.size(); ++i) {
        if (!std::isfinite(first[i]) || !std::isfinite(second[i])) return undefined;
        differences.push_back(static_cast<long double>(second[i])-first[i]);
    }
    return checked(euclideanNorm(differences));
}
