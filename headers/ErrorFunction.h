#ifndef MINIMIZEROPTIMIZER_HEADERS_ERRORFUNCTIONS_H_
#define MINIMIZEROPTIMIZER_HEADERS_ERRORFUNCTIONS_H_

#include "Function.h"
#include <map>
#include <vector>

/*
    ET_POINTSECTIONDIST,
    ET_POINTONSECTION,
    ET_POINTPOINTDIST,
    ET_POINTONPOINT,
    ET_SECTIONCIRCLEDIST,
    ET_SECTIONONCIRCLE,
    ET_SECTIONINCIRCLE,
    ET_SECTIONSECTIONPARALLEL,
    ET_SECTIONSECTIONPERPENDICULAR,
    ET_SECTIONSECTIONANGEL,
    ET_POINTONCIRCLE
 */


class ErrorFunction : public Function {
protected:
    Function *c_f;
    std::vector<Variable *> m_X;
    double v_error;
public:
    ErrorFunction(std::vector<Variable *> x, double error = 0) : c_f(nullptr), m_X(x), v_error(error) {}

	~ErrorFunction() {
        delete c_f;
	}

    std::vector<Variable*> getVariables() {
        return m_X;
    }

    double evaluate() const override{
        return c_f->evaluate();
    }
    std::string to_string() const override {
        return c_f->to_string();
    }

    Function *derivative(Variable *var) const override{
        return c_f->derivative(var);
    }
    Function* simplify() const override {
        return c_f->simplify();
    }
    Function* clone() const override {
        throw std::runtime_error("Not implemented");
    }
};

//1
class PointSectionDistanceError : public ErrorFunction {
public:
    PointSectionDistanceError(std::vector<Variable *> x, double error);
    Function* clone() const override;
};

//2
class PointOnSectionError : public PointSectionDistanceError {
public:
    PointOnSectionError(std::vector<Variable *> x);
    Function* clone() const override;
};

//3
class PointPointDistanceError : public ErrorFunction {
public:
    PointPointDistanceError(std::vector<Variable *> x, double error);
    Function* clone() const override;
};

//4
class PointOnPointError : public PointPointDistanceError {
public:
    PointOnPointError(std::vector<Variable *> x);
    Function* clone() const override;
};

//5
class SectionCircleDistanceError : public ErrorFunction {
public:
    SectionCircleDistanceError(std::vector<Variable *> x, double error);
    Function* clone() const override;
};

//6
class SectionOnCircleError : public SectionCircleDistanceError {
public:
    SectionOnCircleError(std::vector<Variable *> x);
    Function* clone() const override;
};

//7
class SectionInCircleError : public ErrorFunction {
public:
    SectionInCircleError(std::vector<Variable *> x);
    Function* clone() const override;
};

//8
class SectionSectionParallelError : public ErrorFunction {
public:
    SectionSectionParallelError(std::vector<Variable *> x);
    Function* clone() const override;
};

//9
class SectionSectionPerpendicularError : public ErrorFunction {
public:
    SectionSectionPerpendicularError(std::vector<Variable *> x);
    Function* clone() const override;
};

//10
class SectionSectionAngleError : public ErrorFunction {
public:
    SectionSectionAngleError(std::vector<Variable *> x, double error);
    Function* clone() const override;
};

//11
class ArcCenterOnPerpendicularError : public ErrorFunction {
public:
    ArcCenterOnPerpendicularError(std::vector<Variable *> x);
    Function* clone() const override;
};

class VerticalError : public ErrorFunction {
public:
  VerticalError(std::vector<Variable *> x);
  Function* clone() const override;
};

class HorizontalError : public ErrorFunction {
public:
  HorizontalError(std::vector<Variable *> x);
  Function* clone() const override;
};
#endif // ! MINIMIZEROPTIMIZER_HEADERS_ERRORFUNCTIONS_H_
