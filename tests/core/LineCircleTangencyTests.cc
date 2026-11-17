#include "ErrorFunction.h"
#include <gtest/gtest.h>
#include <limits>

TEST(LineCircleTangency, SavedSidesRadiusDerivativeAndReversingTheLine) {
    double ax=0,ay=0,bx=2,by=0,cx=20,cy=3,r=3;
    std::vector<double*> v{&ax,&ay,&bx,&by,&cx,&cy,&r};
    LineCircleTangentError f(v,1);
    EXPECT_TRUE(f.satisfied(0)); EXPECT_EQ(f.gradient().at(&r),-1);
    cy=-3; EXPECT_EQ(f.evaluate(),-6);
    f.setTarget(-1); EXPECT_TRUE(f.satisfied(0)); EXPECT_EQ(f.gradient().at(&r),1);
    std::swap(ax,bx); f.setTarget(1); EXPECT_TRUE(f.satisfied(0));
    f.setWeight(4); cy=-2;
    EXPECT_EQ(f.weightedValue(),-4); EXPECT_EQ(f.weightedGradient().at(&r),-4);
    EXPECT_THROW(f.setTarget(0),std::invalid_argument);
    EXPECT_THROW(LineCircleTangentError(v,2),std::invalid_argument);
    r=0; EXPECT_FALSE(f.satisfied(100));
    EXPECT_THROW(LineCircleTangentError(v,1),std::invalid_argument);
    r=3; bx=ax; by=ay; EXPECT_FALSE(f.satisfied(100));
    f.setWeight(0); EXPECT_EQ(f.weightedValue(),0);
}

TEST(LineCircleTangency, SharedCoordinatesAreDifferentiatedByIdentity) {
    double x=1,y=2,bx=4,by=6,cy=8,r=2;
    LineCircleTangentError f({&x,&y,&bx,&by,&x,&cy,&r},1);
    auto g=f.gradient(); constexpr double h=1e-5;
    x+=h; double plus=f.evaluate(); x-=2*h; double minus=f.evaluate(); x+=h;
    EXPECT_NEAR(g.at(&x),(plus-minus)/(2*h),1e-9);
    auto old=g.at(&x); x+=h; plus=f.gradient().at(&x); x-=2*h;
    minus=f.gradient().at(&x); x+=h;
    EXPECT_NEAR(f.secondDerivative(&x,&x),(plus-minus)/(2*h),1e-8);
    EXPECT_TRUE(std::isfinite(old));
}
