#include "ErrorFunction.h"
#include <gtest/gtest.h>
#include <limits>
#include <memory>
#include <numbers>
#include <functional>
#include <array>
#include <type_traits>

static_assert(std::is_base_of_v<PointSectionDistanceError,PointOnSectionError>);
static_assert(std::is_base_of_v<PointPointDistanceError,PointOnPointError>);
static_assert(std::is_base_of_v<SectionCircleDistanceError,SectionOnCircleError>);

TEST(ConstraintContract, PointDistanceUtilityPreservesDimensionsScalesAndDomainChecks) {
    for (double scale : {1e-300,1e-200,1.0,1e200,1e300}) {
        const std::array<double,3> a{0,0,0}, b{2*scale,3*scale,6*scale};
        EXPECT_NEAR(PointPointDistanceError::distance(a,b)/scale,7,1e-12);
    }
    const std::array<double,1> a{0};
    const std::array<double,2> b{3,4};
    EXPECT_THROW(PointPointDistanceError::distance(a,b),std::invalid_argument);
    const std::array<double,1> invalid{std::numeric_limits<double>::quiet_NaN()};
    EXPECT_FALSE(std::isfinite(PointPointDistanceError::distance(a,invalid)));
}


TEST(ConstraintContract, ConstructionAndParameterChangesValidateTheMathematicalDomain) {
    double x=1, y=2, bx=3, by=4, radius=5;
    const double inf=std::numeric_limits<double>::infinity();
    EXPECT_THROW(FixCoordinateError(std::vector<double*>{}),std::invalid_argument);
    EXPECT_THROW(FixCoordinateError(std::vector<double*>{nullptr}),std::invalid_argument);
    EXPECT_THROW(FixCoordinateError({&x},inf),std::invalid_argument);
    EXPECT_THROW(SectionInCircleError(std::vector<double*>{}),std::logic_error);
    EXPECT_THROW(PointPointDistanceError({&x,&y,&bx,&by},-1),std::invalid_argument);
    EXPECT_THROW(HorizontalError({&x,&y,&bx,&by},1),std::invalid_argument);
    EXPECT_THROW(SectionSectionAngleError({&x,&y,&bx,&by,&x,&y,&bx,&by},4),std::invalid_argument);
    radius=0;
    EXPECT_THROW(PointOnCircleError({&x,&y,&bx,&by,&radius}),std::invalid_argument);
    radius=-1;
    EXPECT_THROW(PointOnCircleError({&x,&y,&bx,&by,&radius}),std::invalid_argument);
    radius=inf;
    EXPECT_THROW(PointOnCircleError({&x,&y,&bx,&by,&radius}),std::invalid_argument);
    FixCoordinateError f({&x},2);
    EXPECT_THROW(f.setWeight(-1),std::invalid_argument);
    EXPECT_THROW(f.setWeight(inf),std::invalid_argument);
    EXPECT_THROW(f.setTarget(inf),std::invalid_argument);
    EXPECT_THROW(f.satisfied(-1),std::invalid_argument);
    EXPECT_THROW(f.derivative(nullptr),std::invalid_argument);
    EXPECT_EQ(f.evaluate(),-1);
    EXPECT_EQ(f.weight(),1);
}

TEST(ConstraintContract, EveryEquationHasIndependentFirstAndSecondDerivativeChecks) {
    const std::vector<std::pair<std::function<std::unique_ptr<ErrorFunction>(std::vector<double*>)>,std::vector<double>>> cases = {
        {[](auto vars) { return std::make_unique<PointPointDistanceError>(vars,0); },{1,2,4,6}}, {[](auto vars) { return std::make_unique<PointOnPointError>(vars,0); },{1,2,4,6}},
        {[](auto vars) { return std::make_unique<PointSectionDistanceError>(vars,0); },{2,7,1,2,5,4}}, {[](auto vars) { return std::make_unique<PointOnSectionError>(vars,0); },{2,7,1,2,5,4}},
        {[](auto vars) { return std::make_unique<SectionCircleDistanceError>(vars,0); },{1,2,5,4,2,7,2}}, {[](auto vars) { return std::make_unique<PointOnCircleError>(vars,0); },{2,7,1,2,2}},
        {[](auto vars) { return std::make_unique<SectionOnCircleError>(vars,0); },{1,2,5,4,2,7,2}},
        {[](auto vars) { return std::make_unique<CircleRadiusError>(vars,5); },{3}},
        {[](auto vars) { return std::make_unique<MidpointCoordinateError>(vars); },{3,1,4}},
        {[](auto vars) { return std::make_unique<EqualLengthError>(vars); },{1,2,4,6,2,7,8,3}},
        {[](auto vars) { return std::make_unique<EqualRadiusError>(vars); },{3,5}},
        {[](auto vars) { return std::make_unique<SectionSectionParallelError>(vars,0); },{1,2,5,4,2,7,8,3}}, {[](auto vars) { return std::make_unique<SectionSectionPerpendicularError>(vars,0); },{1,2,5,4,2,7,8,3}},
        {[](auto vars) { return std::make_unique<SectionSectionAngleError>(vars,1); },{1,2,5,4,2,7,8,3}}, {[](auto vars) { return std::make_unique<VerticalError>(vars,0); },{1,2,5,4}},
        {[](auto vars) { return std::make_unique<HorizontalError>(vars,0); },{1,2,5,4}}, {[](auto vars) { return std::make_unique<ArcCenterOnPerpendicularError>(vars,0); },{1,2,5,4,2,7}}, {[](auto vars) { return std::make_unique<FixCoordinateError>(vars,0); },{3}}
    };
    std::size_t caseIndex=0;
    for (const auto& [create,initial] : cases) {
        SCOPED_TRACE(caseIndex++);
        auto values = initial;
        std::vector<double*> vars;
        for (auto& v : values) vars.push_back(&v);
        auto owner=create(vars);
        auto& f=*owner;
        const auto gradient = f.gradient();
        for (std::size_t i = 0; i < vars.size(); ++i) {
            Variable vi(vars[i]);
            std::unique_ptr<Function> d(f.derivative(&vi));
            constexpr double step = 1e-5;
            const double original = *vars[i];
            *vars[i] = original+step;
            const double plus = f.evaluate();
            *vars[i] = original-step;
            const double minus = f.evaluate();
            *vars[i] = original;
            EXPECT_NEAR(gradient.at(vars[i]),(plus-minus)/(2*step),1e-7);
            EXPECT_NEAR(d->evaluate(),gradient.at(vars[i]),1e-12);
            for (std::size_t j = 0; j < vars.size(); ++j) {
                Variable vj(vars[j]);
                std::unique_ptr<Function> dd(d->derivative(&vj));
                const double old = *vars[j];
                *vars[j] = old+step;
                const double dp = d->evaluate();
                *vars[j] = old-step;
                const double dm = d->evaluate();
                *vars[j] = old;
                EXPECT_NEAR(dd->evaluate(),(dp-dm)/(2*step),2e-7);
                EXPECT_NEAR(dd->evaluate(),f.secondDerivative(vars[i],vars[j]),1e-12);
                EXPECT_NEAR(dd->evaluate(),f.secondDerivative(vars[j],vars[i]),1e-12);
                EXPECT_THROW(dd->derivative(&vi),std::logic_error);
            }
        }
    }
}

TEST(ConstraintContract, DerivativesClonesAndSimplificationRemainLiveAfterOriginalDeletion) {
    double ax=1, ay=2, bx=4, by=6;
    Variable x(&ax), y(&ay);
    auto f = std::make_unique<PointPointDistanceError>(std::vector<double*>{&ax,&ay,&bx,&by},5);
    std::unique_ptr<Function> d(f->derivative(&x)), dd(d->derivative(&y));
    std::unique_ptr<ErrorFunction> clone(f->clone());
    EXPECT_NE(dynamic_cast<PointPointDistanceError*>(clone.get()),nullptr);
    std::unique_ptr<Function> simplified(f->simplify());
    std::unique_ptr<Function> weighted(f->weightedFunction()), dw(weighted->derivative(&x));
    f.reset();
    bx=7;
    EXPECT_NEAR(d->evaluate(),-6/std::hypot(6.0,4.0),1e-12);
    EXPECT_NEAR(clone->evaluate(),simplified->evaluate(),1e-12);
    EXPECT_NEAR(dd->evaluate(),clone->secondDerivative(&ax,&ay),1e-12);
    clone->setWeight(3);
    EXPECT_NEAR(dw->evaluate(),3*d->evaluate(),1e-12);
    clone->setTarget(7);
    EXPECT_NEAR(weighted->evaluate(),3*(std::hypot(6.0,4.0)-7),1e-12);
}

TEST(ConstraintContract, AliasedArgumentsSumFirstAndSecondDerivatives) {
    double shared=2, by=6;
    PointPointDistanceError f({&shared,&shared,&shared,&by});
    EXPECT_NEAR(f.evaluate(),4,1e-12);
    EXPECT_NEAR(f.gradient().at(&shared),-1,1e-12);
    EXPECT_NEAR(f.secondDerivative(&shared,&shared),0,1e-12);
    shared=3;
    EXPECT_NEAR(f.evaluate(),3,1e-12);
    PointOnSectionError line({&shared,&by,&shared,&shared,&by,&by});
    const double original=shared, step=1e-5;
    shared=original+step; const auto plus=line.gradient();
    shared=original-step; const auto minus=line.gradient();
    shared=original;
    EXPECT_NEAR(line.secondDerivative(&shared,&shared),(plus.at(&shared)-minus.at(&shared))/(2*step),1e-7);
}

TEST(ConstraintContract, ScalesDoNotChangeDirectionOrOverflowDistance) {
    for (double scale : {1e-300,1e-200,1e-13,1.0,1e200,1e300}) {
        double ax=0, ay=0, bx=3*scale, by=4*scale, cx=6*scale, cy=8*scale, radius=scale;
        HorizontalError horizontal({&ax,&ay,&bx,&by});
        EXPECT_NEAR(horizontal.evaluate(),0.8,1e-12);
        EXPECT_NEAR(horizontal.gradient().at(&by)*scale,0.072,1e-12);
        PointPointDistanceError distance({&ax,&ay,&bx,&by});
        EXPECT_NEAR(distance.evaluate()/scale,5,1e-12);
        SectionCircleDistanceError clearance({&ax,&ay,&bx,&by,&cx,&cy,&radius});
        EXPECT_NEAR(clearance.evaluate()/scale,4,1e-12);
        EXPECT_TRUE(std::isfinite(clearance.gradient().at(&bx)));
    }
}

TEST(ConstraintContract, OverflowAndInvalidMutationCannotPassAndZeroWeightDisablesThem) {
    double a=-std::numeric_limits<double>::max(), b=std::numeric_limits<double>::max(), zero=0;
    PointPointDistanceError f({&a,&zero,&b,&zero});
    EXPECT_FALSE(f.satisfied(1));
    f.setWeight(0);
    EXPECT_EQ(f.weightedValue(),0);
    for (const auto& [v,d] : f.weightedGradient()) EXPECT_EQ(d,0);
    EXPECT_TRUE(f.satisfied(0));
    a=std::numeric_limits<double>::quiet_NaN();
    EXPECT_TRUE(f.satisfied(0));
    f.setWeight(1);
    EXPECT_FALSE(f.satisfied(1));
    double px=3, py=4, cx=0, cy=0, radius=5;
    PointOnCircleError circle({&px,&py,&cx,&cy,&radius});
    radius=0;
    EXPECT_FALSE(circle.satisfied(100));
}

TEST(ConstraintContract, LegacyErrorsNeverOwnCallerVariablesAndRejectUnsupportedKind) {
    double ax=0, ay=0, bx=4, by=0, cx=2, cy=2, r=2;
    Variable a(&ax), b(&ay), c(&bx), d(&by), e(&cx), f(&cy), g(&r);
    const std::vector<Variable*> vars{&a,&b,&c,&d,&e,&f,&g};
    std::unique_ptr<Function> clone;
    {
        SectionCircleDistanceError error(vars,0);
        EXPECT_NEAR(error.evaluate(),0,1e-12);
        clone.reset(error.clone());
        EXPECT_THROW(SectionInCircleError unsupported(vars),std::logic_error);
    }
    r=1;
    EXPECT_NEAR(clone->evaluate(),1,1e-12);
    EXPECT_NEAR(g.evaluate(),1,1e-12);
}
