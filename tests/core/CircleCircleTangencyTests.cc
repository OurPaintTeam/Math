#include "ErrorFunction.h"
#include "SparseLSMTask.h"
#include "sparse/SparseLevenbergMarquardtSolver.h"
#include <gtest/gtest.h>
#include <limits>

TEST(CircleCircleTangency, BranchesRequireCorrectRadiusOrderAndUniqueInternalContact) {
    double x1=0,y1=0,r1=5,x2=3,y2=0,r2=2;
    std::vector<double*> v{&x1,&y1,&r1,&x2,&y2,&r2};
    CircleCircleTangentError f(v,1);
    EXPECT_TRUE(f.satisfied(0)); EXPECT_EQ(f.gradient().at(&r1),-1); EXPECT_EQ(f.gradient().at(&r2),1);
    f.setTarget(0); x2=7; EXPECT_TRUE(f.satisfied(0));
    EXPECT_EQ(f.gradient().at(&r1),-1); EXPECT_EQ(f.gradient().at(&r2),-1);
    f.setTarget(2); EXPECT_FALSE(f.satisfied(100));
    r1=2;r2=5;x2=3; EXPECT_TRUE(f.satisfied(0));
    EXPECT_EQ(f.gradient().at(&r1),1); EXPECT_EQ(f.gradient().at(&r2),-1);
    x2=0;r1=r2; EXPECT_FALSE(f.satisfied(0));
    f.setWeight(0); EXPECT_TRUE(f.satisfied(0)); f.setWeight(1);
    for (double invalid:{0.0,-1.0,std::numeric_limits<double>::infinity()}) {
        r1=invalid; EXPECT_FALSE(f.satisfied(100));
        EXPECT_THROW(CircleCircleTangentError(v,0),std::invalid_argument);
        r1=5; r2=invalid; EXPECT_FALSE(f.satisfied(100));
        EXPECT_THROW(CircleCircleTangentError(v,0),std::invalid_argument); r2=5;
    }
    EXPECT_THROW(f.setTarget(3),std::invalid_argument);
}

TEST(CircleCircleTangency, SharedCentersAndRadiiSumTheirDerivatives) {
    double x1=0,y=0,x2=7,r=3;
    CircleCircleTangentError f({&x1,&y,&r,&x2,&y,&r},0);
    EXPECT_EQ(f.evaluate(),1); EXPECT_EQ(f.gradient().at(&r),-2);
    EXPECT_EQ(f.gradient().at(&y),0); EXPECT_EQ(f.secondDerivative(&y,&y),0);
    f.setWeight(2); EXPECT_EQ(f.weightedValue(),2); EXPECT_EQ(f.weightedGradient().at(&r),-4);
    EXPECT_EQ(f.secondDerivative(&r,&r),0);
}

TEST(CircleCircleTangency, LMRejectsNonPositiveTrialRadiiAndRestoresResidualCaches) {
    double x1=0,y1=0,r1=5,x2=10,y2=0,r2=1;
    CircleCircleTangentError f({&x1,&y1,&r1,&x2,&y2,&r2},1);
    Variable radius(&r2);
    SparseLSMTask task({f.weightedFunction()},{&radius});
    SparseLMSolver solver(1,1e-3,1e-10,1e-10,1e-14);
    solver.setTask(&task); ASSERT_NO_THROW(solver.optimize());
    EXPECT_GT(f.invalidEvaluations(),0u);
    EXPECT_GT(r2,0); EXPECT_LT(r2,r1);
    EXPECT_EQ(solver.getResult().at(0),r2);
    EXPECT_EQ(task.getError(),solver.getCurrentError());
    EXPECT_NEAR(task.getError(),f.evaluate()*f.evaluate(),1e-12);
    EXPECT_FALSE(solver.isConverged());
}

TEST(CircleCircleTangency, CoincidentCentersEscapeDespiteTheNonzeroRadiusGradient) {
    double x1=0,y1=0,r1=3,x2=0,y2=0,r2=2;
    CircleCircleTangentError f({&x1,&y1,&r1,&x2,&y2,&r2},0);
    CircleRadiusError size({&r2},2);
    Variable x(&x2),y(&y2),radius(&r2);
    SparseLSMTask task({f.weightedFunction(),size.weightedFunction()},{&x,&y,&radius});
    EXPECT_EQ(task.normalGradient()(0,0),0);
    EXPECT_EQ(task.normalGradient()(1,0),0);
    EXPECT_NE(task.normalGradient()(2,0),0);
    SparseLMSolver solver(300,1e-3,1e-10,1e-10,1e-14);
    for (int attempt=0;attempt<2;++attempt) {
        x2=0;y2=0;r2=2;
        solver.setTask(&task); solver.optimize();
        ASSERT_TRUE(solver.isConverged());
        EXPECT_NEAR(std::hypot(x2,y2),5,1e-6);
        EXPECT_NEAR(r2,2,1e-6);
    }
}
