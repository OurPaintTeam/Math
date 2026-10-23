#pragma once
#include "Function.h"
#include <memory>
#include <span>
#include <unordered_map>
#include <vector>

// Existing mathematical API. Coordinates and caller Variable wrappers are borrowed.
// Copies and derivatives share live coordinates, targets and weights.
class ErrorFunction : public Function {
protected:
    enum class Equation {
        PointLineDistance, PointOnLine, PointPointDistance, PointOnPoint,
        SegmentCircleDistance, PointOnCircle, SegmentOnCircle,
        Parallel, Perpendicular, Angle, Vertical, Horizontal, ArcBisector,
        FixCoordinate, SegmentInCircle, CircleRadius, EqualLength, EqualRadius, MidpointCoordinate,
        SymmetryAlong, SymmetryAcross, CoordinateDifference, CoordinateAverage, LineCircleTangent,
        CircleCircleTangent, ArcLineTangent, ArcArcTangent, CircleArcTangent, PointPointHorizontalDistance, PointPointVerticalDistance, PointOnArc, ArcMidpointX, ArcMidpointY, MidpointDifference, ArcArcMidpointX, ArcArcMidpointY, LineArcMidpointX, LineArcMidpointY
    };
    ErrorFunction(Equation equation, std::vector<double*> coordinates, double target);
    static std::vector<double*> coordinatePointers(const std::vector<Variable*>& variables);
private:
    struct State;
    std::shared_ptr<State> _state;
    std::vector<Variable> _variables;
    std::vector<double*> _withRespectTo;
    bool _weighted = false;
public:
    using Gradient = std::unordered_map<double*, double>;
    ErrorFunction(const ErrorFunction&) = default;
    ErrorFunction& operator=(const ErrorFunction&) = default;
    std::vector<Variable*> getVariables();
    const std::vector<double*>& variables() const;
    double evaluate() const override;
    virtual Gradient gradient() const;
    double secondDerivative(double* first, double* second) const;
    Function* derivative(Variable* variable) const override;
    ErrorFunction* clone() const override;
    Function* simplify() const override { return clone(); }
    std::string to_string() const override;
    std::vector<double*> referencedCoordinates() const override { return variables(); }
    std::size_t revision() const override;
    std::size_t invalidEvaluations() const;
    bool assignment(double*& coordinate, double& target) const;
    void setTarget(double target);
    double weight() const;
    void setWeight(double weight);
    virtual double weightedValue() const;
    virtual Gradient weightedGradient() const;
    virtual Function* weightedFunction() const;
    bool satisfied(double tolerance) const;
};

class PointSectionDistanceError : public ErrorFunction {
protected:
    PointSectionDistanceError(Equation equation, std::vector<double*> coordinates, double target)
        : ErrorFunction(equation,std::move(coordinates),target) {}
public:
    explicit PointSectionDistanceError(std::vector<Variable*> variables, double target = 0)
        : PointSectionDistanceError(coordinatePointers(variables),target) {}
    explicit PointSectionDistanceError(std::vector<double*> coordinates, double target = 0)
        : ErrorFunction(Equation::PointLineDistance,std::move(coordinates),target) {}
    PointSectionDistanceError* clone() const override { return new PointSectionDistanceError(*this); }
};

class PointOnSectionError : public PointSectionDistanceError {
public:
    explicit PointOnSectionError(std::vector<Variable*> variables, double target = 0)
        : PointOnSectionError(coordinatePointers(variables),target) {}
    explicit PointOnSectionError(std::vector<double*> coordinates, double target = 0)
        : PointSectionDistanceError(Equation::PointOnLine,std::move(coordinates),target) {}
    PointOnSectionError* clone() const override { return new PointOnSectionError(*this); }
};

class PointPointDistanceError : public ErrorFunction {
protected:
    PointPointDistanceError(Equation equation, std::vector<double*> coordinates, double target)
        : ErrorFunction(equation,std::move(coordinates),target) {}
public:
    static double distance(std::span<const double> first, std::span<const double> second);
    explicit PointPointDistanceError(std::vector<Variable*> variables, double target = 0)
        : PointPointDistanceError(coordinatePointers(variables),target) {}
    explicit PointPointDistanceError(std::vector<double*> coordinates, double target = 0)
        : ErrorFunction(Equation::PointPointDistance,std::move(coordinates),target) {}
    PointPointDistanceError* clone() const override { return new PointPointDistanceError(*this); }
};

class PointOnPointError : public PointPointDistanceError {
public:
    explicit PointOnPointError(std::vector<Variable*> variables, double target = 0)
        : PointOnPointError(coordinatePointers(variables),target) {}
    explicit PointOnPointError(std::vector<double*> coordinates, double target = 0)
        : PointPointDistanceError(Equation::PointOnPoint,std::move(coordinates),target) {}
    PointOnPointError* clone() const override { return new PointOnPointError(*this); }
};

class SectionCircleDistanceError : public ErrorFunction {
protected:
    SectionCircleDistanceError(Equation equation, std::vector<double*> coordinates, double target)
        : ErrorFunction(equation,std::move(coordinates),target) {}
public:
    explicit SectionCircleDistanceError(std::vector<Variable*> variables, double target = 0)
        : SectionCircleDistanceError(coordinatePointers(variables),target) {}
    explicit SectionCircleDistanceError(std::vector<double*> coordinates, double target = 0)
        : ErrorFunction(Equation::SegmentCircleDistance,std::move(coordinates),target) {}
    SectionCircleDistanceError* clone() const override { return new SectionCircleDistanceError(*this); }
};

// Supporting infinite line A,B and circle C,r. Side is +1 (left of A->B) or -1.
class LineCircleTangentError : public ErrorFunction {
public:
    LineCircleTangentError(std::vector<Variable*> variables, double side)
        : LineCircleTangentError(coordinatePointers(variables),side) {}
    LineCircleTangentError(std::vector<double*> coordinates, double side)
        : ErrorFunction(Equation::LineCircleTangent,std::move(coordinates),side) {}
    LineCircleTangentError* clone() const override { return new LineCircleTangentError(*this); }
};

// Arguments C1x,C1y,r1,C2x,C2y,r2. Kind: 0 external, 1 first contains second, 2 reverse.
// Internal branches require a strictly larger containing radius (unique contact).
class CircleCircleTangentError : public ErrorFunction {
public:
    CircleCircleTangentError(std::vector<Variable*> variables, double kind)
        : CircleCircleTangentError(coordinatePointers(variables),kind) {}
    CircleCircleTangentError(std::vector<double*> coordinates, double kind)
        : ErrorFunction(Equation::CircleCircleTangent,std::move(coordinates),kind) {}
    CircleCircleTangentError* clone() const override { return new CircleCircleTangentError(*this); }
};

// Radius point P, arc center C, line A,B (xy pairs). Uses the supporting circle.
class ArcLineTangentError : public ErrorFunction {
public:
    explicit ArcLineTangentError(std::vector<Variable*> variables, double target)
        : ArcLineTangentError(coordinatePointers(variables),target) {}
    explicit ArcLineTangentError(std::vector<double*> coordinates, double target)
        : ErrorFunction(Equation::ArcLineTangent,std::move(coordinates),target) {}
    ArcLineTangentError* clone() const override { return new ArcLineTangentError(*this); }
};
// Radius point P1, center C1, radius point P2, center C2 (xy pairs).
class ArcArcTangentError : public ErrorFunction {
public:
    explicit ArcArcTangentError(std::vector<Variable*> variables, double target)
        : ArcArcTangentError(coordinatePointers(variables),target) {}
    explicit ArcArcTangentError(std::vector<double*> coordinates, double target)
        : ErrorFunction(Equation::ArcArcTangent,std::move(coordinates),target) {}
    ArcArcTangentError* clone() const override { return new ArcArcTangentError(*this); }
};

// Circle Cx,Cy,r followed by arc Px,Py,Cx,Cy; kind has the same ordered meaning.
class CircleArcTangentError : public ErrorFunction {
public:
    CircleArcTangentError(std::vector<Variable*> variables, double kind)
        : CircleArcTangentError(coordinatePointers(variables),kind) {}
    CircleArcTangentError(std::vector<double*> coordinates, double kind)
        : ErrorFunction(Equation::CircleArcTangent,std::move(coordinates),kind) {}
    CircleArcTangentError* clone() const override { return new CircleArcTangentError(*this); }
};


class PointOnCircleError : public ErrorFunction {
public:
    explicit PointOnCircleError(std::vector<Variable*> variables, double target = 0)
        : PointOnCircleError(coordinatePointers(variables),target) {}
    explicit PointOnCircleError(std::vector<double*> coordinates, double target = 0)
        : ErrorFunction(Equation::PointOnCircle,std::move(coordinates),target) {}
    PointOnCircleError* clone() const override { return new PointOnCircleError(*this); }
};

class CircleRadiusError : public ErrorFunction {
public:
    CircleRadiusError(std::vector<Variable*> variables, double target)
        : CircleRadiusError(coordinatePointers(variables),target) {}
    CircleRadiusError(std::vector<double*> coordinates, double target)
        : ErrorFunction(Equation::CircleRadius,std::move(coordinates),target) {}
    CircleRadiusError* clone() const override { return new CircleRadiusError(*this); }
};

// One independent coordinate of P = (A + B)/2; arguments are P, A, B.
class MidpointCoordinateError : public ErrorFunction {
public:
    explicit MidpointCoordinateError(std::vector<Variable*> variables, double target = 0)
        : MidpointCoordinateError(coordinatePointers(variables),target) {}
    explicit MidpointCoordinateError(std::vector<double*> coordinates, double target = 0)
        : ErrorFunction(Equation::MidpointCoordinate,std::move(coordinates),target) {}
    MidpointCoordinateError* clone() const override { return new MidpointCoordinateError(*this); }
};

// Arguments P, Q, A, B (xy pairs); A and B define the symmetry axis.
class SymmetryAlongError : public ErrorFunction {
public:
    explicit SymmetryAlongError(std::vector<Variable*> variables, double target = 0)
        : SymmetryAlongError(coordinatePointers(variables),target) {}
    explicit SymmetryAlongError(std::vector<double*> coordinates, double target = 0)
        : ErrorFunction(Equation::SymmetryAlong,std::move(coordinates),target) {}
    SymmetryAlongError* clone() const override { return new SymmetryAlongError(*this); }
};
class SymmetryAcrossError : public ErrorFunction {
public:
    explicit SymmetryAcrossError(std::vector<Variable*> variables, double target = 0)
        : SymmetryAcrossError(coordinatePointers(variables),target) {}
    explicit SymmetryAcrossError(std::vector<double*> coordinates, double target = 0)
        : ErrorFunction(Equation::SymmetryAcross,std::move(coordinates),target) {}
    SymmetryAcrossError* clone() const override { return new SymmetryAcrossError(*this); }
};
// Two coordinates: equality and average equal to the prescribed axis offset.
class CoordinateDifferenceError : public ErrorFunction {
public:
    explicit CoordinateDifferenceError(std::vector<Variable*> variables, double target = 0)
        : CoordinateDifferenceError(coordinatePointers(variables),target) {}
    explicit CoordinateDifferenceError(std::vector<double*> coordinates, double target = 0)
        : ErrorFunction(Equation::CoordinateDifference,std::move(coordinates),target) {}
    CoordinateDifferenceError* clone() const override { return new CoordinateDifferenceError(*this); }
};
class CoordinateAverageError : public ErrorFunction {
public:
    explicit CoordinateAverageError(std::vector<Variable*> variables, double target = 0)
        : CoordinateAverageError(coordinatePointers(variables),target) {}
    explicit CoordinateAverageError(std::vector<double*> coordinates, double target = 0)
        : ErrorFunction(Equation::CoordinateAverage,std::move(coordinates),target) {}
    CoordinateAverageError* clone() const override { return new CoordinateAverageError(*this); }
};

class EqualLengthError : public ErrorFunction {
public:
    explicit EqualLengthError(std::vector<Variable*> variables, double target = 0)
        : EqualLengthError(coordinatePointers(variables),target) {}
    explicit EqualLengthError(std::vector<double*> coordinates, double target = 0)
        : ErrorFunction(Equation::EqualLength,std::move(coordinates),target) {}
    EqualLengthError* clone() const override { return new EqualLengthError(*this); }
};

class EqualRadiusError : public ErrorFunction {
public:
    explicit EqualRadiusError(std::vector<Variable*> variables, double target = 0)
        : EqualRadiusError(coordinatePointers(variables),target) {}
    explicit EqualRadiusError(std::vector<double*> coordinates, double target = 0)
        : ErrorFunction(Equation::EqualRadius,std::move(coordinates),target) {}
    EqualRadiusError* clone() const override { return new EqualRadiusError(*this); }
};

class SectionOnCircleError : public SectionCircleDistanceError {
public:
    explicit SectionOnCircleError(std::vector<Variable*> variables, double target = 0)
        : SectionOnCircleError(coordinatePointers(variables),target) {}
    explicit SectionOnCircleError(std::vector<double*> coordinates, double target = 0)
        : SectionCircleDistanceError(Equation::SegmentOnCircle,std::move(coordinates),target) {}
    SectionOnCircleError* clone() const override { return new SectionOnCircleError(*this); }
};

class SectionSectionParallelError : public ErrorFunction {
public:
    explicit SectionSectionParallelError(std::vector<Variable*> variables, double target = 0)
        : SectionSectionParallelError(coordinatePointers(variables),target) {}
    explicit SectionSectionParallelError(std::vector<double*> coordinates, double target = 0)
        : ErrorFunction(Equation::Parallel,std::move(coordinates),target) {}
    SectionSectionParallelError* clone() const override { return new SectionSectionParallelError(*this); }
};

class SectionSectionPerpendicularError : public ErrorFunction {
public:
    explicit SectionSectionPerpendicularError(std::vector<Variable*> variables, double target = 0)
        : SectionSectionPerpendicularError(coordinatePointers(variables),target) {}
    explicit SectionSectionPerpendicularError(std::vector<double*> coordinates, double target = 0)
        : ErrorFunction(Equation::Perpendicular,std::move(coordinates),target) {}
    SectionSectionPerpendicularError* clone() const override { return new SectionSectionPerpendicularError(*this); }
};

class SectionSectionAngleError : public ErrorFunction {
public:
    explicit SectionSectionAngleError(std::vector<Variable*> variables, double target = 0)
        : SectionSectionAngleError(coordinatePointers(variables),target) {}
    explicit SectionSectionAngleError(std::vector<double*> coordinates, double target = 0)
        : ErrorFunction(Equation::Angle,std::move(coordinates),target) {}
    SectionSectionAngleError* clone() const override { return new SectionSectionAngleError(*this); }
};

class VerticalError : public ErrorFunction {
public:
    explicit VerticalError(std::vector<Variable*> variables, double target = 0)
        : VerticalError(coordinatePointers(variables),target) {}
    explicit VerticalError(std::vector<double*> coordinates, double target = 0)
        : ErrorFunction(Equation::Vertical,std::move(coordinates),target) {}
    VerticalError* clone() const override { return new VerticalError(*this); }
};

class HorizontalError : public ErrorFunction {
public:
    explicit HorizontalError(std::vector<Variable*> variables, double target = 0)
        : HorizontalError(coordinatePointers(variables),target) {}
    explicit HorizontalError(std::vector<double*> coordinates, double target = 0)
        : ErrorFunction(Equation::Horizontal,std::move(coordinates),target) {}
    HorizontalError* clone() const override { return new HorizontalError(*this); }
};

class ArcCenterOnPerpendicularError : public ErrorFunction {
public:
    explicit ArcCenterOnPerpendicularError(std::vector<Variable*> variables, double target = 0)
        : ArcCenterOnPerpendicularError(coordinatePointers(variables),target) {}
    explicit ArcCenterOnPerpendicularError(std::vector<double*> coordinates, double target = 0)
        : ErrorFunction(Equation::ArcBisector,std::move(coordinates),target) {}
    ArcCenterOnPerpendicularError* clone() const override { return new ArcCenterOnPerpendicularError(*this); }
};

class FixCoordinateError : public ErrorFunction {
public:
    explicit FixCoordinateError(std::vector<Variable*> variables, double target = 0)
        : FixCoordinateError(coordinatePointers(variables),target) {}
    explicit FixCoordinateError(std::vector<double*> coordinates, double target = 0)
        : ErrorFunction(Equation::FixCoordinate,std::move(coordinates),target) {}
    FixCoordinateError* clone() const override { return new FixCoordinateError(*this); }
};

class SectionInCircleError : public ErrorFunction {
public:
    explicit SectionInCircleError(std::vector<Variable*> variables, double target = 0)
        : SectionInCircleError(coordinatePointers(variables),target) {}
    explicit SectionInCircleError(std::vector<double*> coordinates, double target = 0)
        : ErrorFunction(Equation::SegmentInCircle,std::move(coordinates),target) {}
    SectionInCircleError* clone() const override { return new SectionInCircleError(*this); }
};

// Coordinates: first point X/Y, second point X/Y. Target is non-negative.
class PointPointHorizontalDistanceError : public ErrorFunction {
public:
    explicit PointPointHorizontalDistanceError(std::vector<Variable*> variables, double target = 0)
        : PointPointHorizontalDistanceError(coordinatePointers(variables), target) {}
    explicit PointPointHorizontalDistanceError(std::vector<double*> coordinates, double target = 0)
        : ErrorFunction(Equation::PointPointHorizontalDistance, std::move(coordinates), target) {}
    PointPointHorizontalDistanceError* clone() const override { return new PointPointHorizontalDistanceError(*this); }
};

// Coordinates: first point X/Y, second point X/Y. Target is non-negative.
class PointPointVerticalDistanceError : public ErrorFunction {
public:
    explicit PointPointVerticalDistanceError(std::vector<Variable*> variables, double target = 0)
        : PointPointVerticalDistanceError(coordinatePointers(variables), target) {}
    explicit PointPointVerticalDistanceError(std::vector<double*> coordinates, double target = 0)
        : ErrorFunction(Equation::PointPointVerticalDistance, std::move(coordinates), target) {}
    PointPointVerticalDistanceError* clone() const override { return new PointPointVerticalDistanceError(*this); }
};

// Point X/Y, first arc endpoint X/Y, center X/Y; membership in the supporting circle.
class PointOnArcError : public ErrorFunction {
public:
    explicit PointOnArcError(std::vector<Variable*> variables, double target = 0)
        : PointOnArcError(coordinatePointers(variables), target) {}
    explicit PointOnArcError(std::vector<double*> coordinates, double target = 0)
        : ErrorFunction(Equation::PointOnArc, std::move(coordinates), target) {}
    PointOnArcError* clone() const override { return new PointOnArcError(*this); }
};

// Point, first endpoint, second endpoint, center (X/Y pairs), counterclockwise arc.
class ArcMidpointXError : public ErrorFunction {
public:
    explicit ArcMidpointXError(std::vector<Variable*> variables, double target = 0)
        : ArcMidpointXError(coordinatePointers(variables), target) {}
    explicit ArcMidpointXError(std::vector<double*> coordinates, double target = 0)
        : ErrorFunction(Equation::ArcMidpointX, std::move(coordinates), target) {}
    ArcMidpointXError* clone() const override { return new ArcMidpointXError(*this); }
};

// Point, first endpoint, second endpoint, center (X/Y pairs), counterclockwise arc.
class ArcMidpointYError : public ErrorFunction {
public:
    explicit ArcMidpointYError(std::vector<Variable*> variables, double target = 0)
        : ArcMidpointYError(coordinatePointers(variables), target) {}
    explicit ArcMidpointYError(std::vector<double*> coordinates, double target = 0)
        : ErrorFunction(Equation::ArcMidpointY, std::move(coordinates), target) {}
    ArcMidpointYError* clone() const override { return new ArcMidpointYError(*this); }
};

// One coordinate of each endpoint: first line A/B, second line C/D.
class MidpointDifferenceError : public ErrorFunction {
public:
    explicit MidpointDifferenceError(std::vector<Variable*> variables, double target = 0)
        : MidpointDifferenceError(coordinatePointers(variables), target) {}
    explicit MidpointDifferenceError(std::vector<double*> coordinates, double target = 0)
        : ErrorFunction(Equation::MidpointDifference, std::move(coordinates), target) {}
    MidpointDifferenceError* clone() const override { return new MidpointDifferenceError(*this); }
};

// First arc A/B/center, second arc A/B/center (X/Y pairs).
class ArcArcMidpointXError : public ErrorFunction {
public:
    explicit ArcArcMidpointXError(std::vector<Variable*> variables, double target = 0)
        : ArcArcMidpointXError(coordinatePointers(variables), target) {}
    explicit ArcArcMidpointXError(std::vector<double*> coordinates, double target = 0)
        : ErrorFunction(Equation::ArcArcMidpointX, std::move(coordinates), target) {}
    ArcArcMidpointXError* clone() const override { return new ArcArcMidpointXError(*this); }
};

// First arc A/B/center, second arc A/B/center (X/Y pairs).
class ArcArcMidpointYError : public ErrorFunction {
public:
    explicit ArcArcMidpointYError(std::vector<Variable*> variables, double target = 0)
        : ArcArcMidpointYError(coordinatePointers(variables), target) {}
    explicit ArcArcMidpointYError(std::vector<double*> coordinates, double target = 0)
        : ErrorFunction(Equation::ArcArcMidpointY, std::move(coordinates), target) {}
    ArcArcMidpointYError* clone() const override { return new ArcArcMidpointYError(*this); }
};

// Line A/B, arc A/B/center (X/Y pairs).
class LineArcMidpointXError : public ErrorFunction {
public:
    explicit LineArcMidpointXError(std::vector<Variable*> variables, double target = 0)
        : LineArcMidpointXError(coordinatePointers(variables), target) {}
    explicit LineArcMidpointXError(std::vector<double*> coordinates, double target = 0)
        : ErrorFunction(Equation::LineArcMidpointX, std::move(coordinates), target) {}
    LineArcMidpointXError* clone() const override { return new LineArcMidpointXError(*this); }
};

// Line A/B, arc A/B/center (X/Y pairs).
class LineArcMidpointYError : public ErrorFunction {
public:
    explicit LineArcMidpointYError(std::vector<Variable*> variables, double target = 0)
        : LineArcMidpointYError(coordinatePointers(variables), target) {}
    explicit LineArcMidpointYError(std::vector<double*> coordinates, double target = 0)
        : ErrorFunction(Equation::LineArcMidpointY, std::move(coordinates), target) {}
    LineArcMidpointYError* clone() const override { return new LineArcMidpointYError(*this); }
};
