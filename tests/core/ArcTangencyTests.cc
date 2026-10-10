#include "ErrorFunction.h"
#include <gtest/gtest.h>

TEST(ArcTangency, ArcLineAcceptsBothLineDirectionsAndRejectsZeroRadiusOrLength) {
    double tx=1,ty=0,cx=0,cy=0,bx=1,by=3;
    // The first line endpoint and arc endpoint share their coordinates.
    ArcLineTangentError f({&tx,&ty,&cx,&cy,&tx,&ty,&bx,&by});
    EXPECT_TRUE(f.satisfied(0)); by=-3; EXPECT_TRUE(f.satisfied(0));
    bx=2; const auto g=f.gradient(); constexpr double h=1e-5;
    for (double* p:{&tx,&ty}) {
        const double original=*p;
        *p=original+h; const double plus=f.evaluate(); *p=original-h;
        const double minus=f.evaluate(); *p=original;
        EXPECT_NEAR(g.at(p),(plus-minus)/(2*h),1e-9);
    }
    bx=tx;by=ty; EXPECT_FALSE(f.satisfied(1));
    bx=2;by=3;cx=tx;cy=ty; EXPECT_FALSE(f.satisfied(1));
    f.setWeight(0); EXPECT_TRUE(f.satisfied(0));
    EXPECT_THROW(f.setTarget(1),std::invalid_argument);
}

TEST(ArcTangency, ArcArcAcceptsCollinearRadiiInBothDirectionsAndRejectsEitherZeroRadius) {
    double tx=1,ty=0,c1x=0,c1y=0,c2x=3,c2y=0;
    ArcArcTangentError f({&tx,&ty,&c1x,&c1y,&c2x,&c2y});
    EXPECT_TRUE(f.satisfied(0)); c2x=-2; EXPECT_TRUE(f.satisfied(0));
    c2y=2; f.setWeight(3);
    EXPECT_EQ(f.evaluate(),-2); EXPECT_EQ(f.weightedValue(),-6);
    EXPECT_EQ(f.weightedGradient().at(&c2y),-3);
    c2x=tx;c2y=ty; EXPECT_FALSE(f.satisfied(1));
    c2x=3;c1x=tx;c1y=ty; EXPECT_FALSE(f.satisfied(1));
    EXPECT_THROW(f.setTarget(1),std::invalid_argument);
}

TEST(ArcTangency, SharedCentersAccumulateBothGradientAndHessianContributions) {
    double tx=2,ty=1,cx=0,cy=0;
    ArcArcTangentError f({&tx,&ty,&cx,&cy,&cx,&cy});
    EXPECT_TRUE(f.satisfied(0));
    for (auto* p:{&tx,&ty,&cx,&cy}) {
        EXPECT_NEAR(f.gradient().at(p),0,1e-12);
        for (auto* q:{&tx,&ty,&cx,&cy}) EXPECT_NEAR(f.secondDerivative(p,q),0,1e-12);
    }
}
