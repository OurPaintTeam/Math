#include "ErrorFunction.h"
#include "SparseLSMTask.h"
#include <gtest/gtest.h>


TEST(ConstraintSystem, WeightsDomainAndCoordinateChangesUseTheSameRows) {
    double x=2;
    auto f=std::make_shared<FixCoordinateError>(std::vector<double*>{&x},1);
    Variable variable(&x);
    f->setWeight(3);
    SparseLSMTask task({f->weightedFunction()},{&variable});
    EXPECT_EQ(task.residualVector()[0],3);
    EXPECT_EQ(task.J().coeff(0,0),3);
    EXPECT_EQ(task.JTJ().coeff(0,0),9);
    EXPECT_EQ(task.diagnose(),SparseLSMTask::DiagnosticStatus::WELL_CONSTRAINED);
    f->setWeight(0);
    EXPECT_EQ(task.residualVector()[0],0);
    EXPECT_EQ(task.J().coeff(0,0),0);
    EXPECT_EQ(task.diagnose(),SparseLSMTask::DiagnosticStatus::EMPTY);
}

TEST(ConstraintSystem, SparseTaskObservesLockedCoordinatesTargetsAndWeightsInCachedRuns) {
    double ax=0, ay=0, bx=3, by=4;
    auto f=std::make_shared<PointPointDistanceError>(std::vector<double*>{&ax,&ay,&bx,&by},5);
    Variable x(&bx);
    SparseLSMTask task({f->weightedFunction()},{&x});
    EXPECT_NEAR(task.residuals()(0,0),0,1e-12);
    EXPECT_NEAR(task.jacobian().coeff(0,0),0.6,1e-12);
    // by is deliberately omitted from optimized variables.
    by=0;
    EXPECT_NEAR(task.residuals()(0,0),-2,1e-12);
    EXPECT_NEAR(task.jacobian().coeff(0,0),1,1e-12);
    f->setWeight(3);
    EXPECT_NEAR(task.residuals()(0,0),-6,1e-12);
    EXPECT_NEAR(task.jacobian().coeff(0,0),3,1e-12);
    f->setTarget(2);
    EXPECT_NEAR(task.residuals()(0,0),3,1e-12);
    EXPECT_NEAR(task.hessian()(0,0),18,1e-12);
    f->setWeight(0);
    EXPECT_EQ(task.getError(),0);
    EXPECT_EQ(task.jacobian().coeff(0,0),0);
}

TEST(ConstraintSystem, BothEndpointRowsRetainIndependentErrors) {
    double ax=3, ay=0, bx=7, by=0, cx=0, cy=0, r=5;
    auto a=std::make_unique<PointOnCircleError>(std::vector<double*>{&ax,&ay,&cx,&cy,&r});
    auto b=std::make_unique<PointOnCircleError>(std::vector<double*>{&bx,&by,&cx,&cy,&r});
    Variable x1(&ax),y1(&ay),x2(&bx),y2(&by),x3(&cx),y3(&cy),radius(&r);
    SparseLSMTask task({a->weightedFunction(),b->weightedFunction()},{&x1,&y1,&x2,&y2,&x3,&y3,&radius});
    EXPECT_EQ(task.residualVector()[0],-2);
    EXPECT_EQ(task.residualVector()[1],2);
    EXPECT_EQ(task.J().rows(),2);
    EXPECT_EQ(task.residualVector().squaredNorm(),8);
}

TEST(ConstraintSystem, RankDiagnosisIgnoresStructuralZeroEntriesInTheCachedJacobian) {
    double ax=0,ay=0,bx=1,by=0;
    HorizontalError horizontal(std::vector<double*>{&ax,&ay,&bx,&by});
    Variable x1(&ax),y1(&ay),x2(&bx),y2(&by);
    SparseLSMTask task({horizontal.weightedFunction()},{&x1,&y1,&x2,&y2});
    EXPECT_EQ(task.J().coeff(0,0),0);
    EXPECT_EQ(task.J().coeff(0,1),-1);
    EXPECT_EQ(task.diagnose(),SparseLSMTask::DiagnosticStatus::UNDER_CONSTRAINED);
}

namespace {
class TrackedExpression final : public Function {
    int& live;
    bool rejectDerivative;
public:
    TrackedExpression(int& live, bool reject = false) : live(live), rejectDerivative(reject) { ++live; }
    ~TrackedExpression() override { --live; }
    double evaluate() const override { return 1; }
    Function* derivative(Variable*) const override {
        if (rejectDerivative) throw std::logic_error("unsupported derivative");
        return clone();
    }
    Function* clone() const override { return new TrackedExpression(live,rejectDerivative); }
    std::string to_string() const override { return "tracked expression"; }
};
}

TEST(ConstraintSystem, SparseTaskReleasesFunctionsAndPartialDerivativesWhenConstructionThrows) {
    int live=0;
    double value=2;
    Variable variable(&value);
    EXPECT_THROW(SparseLSMTask task(
        {new TrackedExpression(live),new TrackedExpression(live,true)},{&variable}),std::logic_error);
    EXPECT_EQ(live,0);
    EXPECT_EQ(variable.evaluate(),2);
}

TEST(ConstraintSystem, SparseTaskObservesExcludedMinMaxBranchInputs) {
    double a=3,b=2;
    Variable x(&a),y(&b);
    Max maximum(x.clone(),y.clone());
    SparseLSMTask task({maximum.derivative(&x)},{&x});
    EXPECT_EQ(task.residuals()(0,0),1);
    b=4;
    EXPECT_EQ(task.residuals()(0,0),0);
}
