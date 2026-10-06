#pragma once
#include "ErrorFunction.h"

// Test-only numerical failure injection. No production geometry equation lives here.
class FaultInjectionConstraint final : public PointPointDistanceError {
    double* _coordinate;
    double _initial, _invalidValue;
    bool _throw;
public:
    FaultInjectionConstraint(double* coordinate, double invalidValue, bool throwOnCandidate)
        : PointPointDistanceError(std::vector<double*>{coordinate,coordinate,coordinate,coordinate}),
          _coordinate(coordinate), _initial(*coordinate), _invalidValue(invalidValue), _throw(throwOnCandidate) {}
    double evaluate() const override {
        if (*_coordinate != _initial) {
            *_coordinate = _invalidValue;
            if (_throw) throw std::runtime_error("Injected optimizer evaluation failure");
            return 0;
        }
        return -1;
    }
    Gradient gradient() const override { return {{_coordinate,1}}; }
    double weightedValue() const override { return weight() == 0 ? 0 : weight()*evaluate(); }
    Gradient weightedGradient() const override { return {{_coordinate,weight()}}; }
    ::Function* derivative(Variable* variable) const override { return new Constant(variable->value == _coordinate ? weight() : 0); }
    FaultInjectionConstraint* clone() const override { return new FaultInjectionConstraint(*this); }
    ::Function* weightedFunction() const override { return clone(); }
};
