#include "ErrorFunction.h"
#include <gtest/gtest.h>
#include <cmath>

TEST(ArcTangency, ArcLineUsesCircleEquationForBothSidesAndArbitraryRadiusPoints) {
    double px=3,py=4,cx=0,cy=0,ax=-7,ay=-5,bx=9,by=-5,r=5;
    for (double side : {-1.,1.}) {
        ArcLineTangentError arc({&px,&py,&cx,&cy,&ax,&ay,&bx,&by},side);
        LineCircleTangentError circle({&ax,&ay,&bx,&by,&cx,&cy,&r},side);
        EXPECT_DOUBLE_EQ(arc.evaluate(),circle.evaluate());
        EXPECT_EQ(arc.satisfied(0),side==1);
        px=-4;py=3; EXPECT_DOUBLE_EQ(arc.evaluate(),circle.evaluate());
        px=6;py=8;r=10; EXPECT_DOUBLE_EQ(arc.evaluate(),circle.evaluate());
        px=3;py=4;r=5;
        EXPECT_THROW(arc.setTarget(0),std::invalid_argument);
    }
    ArcLineTangentError arc({&px,&py,&cx,&cy,&ax,&ay,&bx,&by},1);
    bx=ax;by=ay; EXPECT_FALSE(arc.satisfied(1));
    bx=9;px=cx;py=cy; EXPECT_FALSE(arc.satisfied(1));
    arc.setWeight(0); EXPECT_TRUE(arc.satisfied(0));
}

TEST(ArcTangency, ArcArcAndMixedUseCircleEquationForEveryBranch) {
    for (double kind : {0.,1.,2.}) {
        double c1x=0,c1y=0,c2x=8,c2y=0;
        double r1=kind==2 ? 3 : 5,r2=kind==2 ? 5 : 3;
        double p1x=0,p1y=r1,p2x=c2x,p2y=-r2;
        ArcArcTangentError arcs({&p1x,&p1y,&c1x,&c1y,&p2x,&p2y,&c2x,&c2y},kind);
        CircleArcTangentError mixed({&c1x,&c1y,&r1,&p2x,&p2y,&c2x,&c2y},kind);
        CircleCircleTangentError circles({&c1x,&c1y,&r1,&c2x,&c2y,&r2},kind);
        EXPECT_DOUBLE_EQ(arcs.evaluate(),circles.evaluate());
        EXPECT_DOUBLE_EQ(mixed.evaluate(),circles.evaluate());
        c2x=kind==0 ? r1+r2 : std::abs(r1-r2);p2x=c2x;
        EXPECT_TRUE(arcs.satisfied(0)); EXPECT_TRUE(mixed.satisfied(0));
        // Rotating the radius point does not select a new contact point.
        p2x=c2x+r2;p2y=0; EXPECT_TRUE(arcs.satisfied(0));
        p2x+=1;r2+=1;
        EXPECT_DOUBLE_EQ(arcs.evaluate(),circles.evaluate());
        EXPECT_DOUBLE_EQ(mixed.evaluate(),circles.evaluate());
        p2x=c2x;p2y=c2y; EXPECT_FALSE(arcs.satisfied(1)); EXPECT_FALSE(mixed.satisfied(1));
        p2x=c2x+2;p1x=c1x;p1y=c1y; EXPECT_FALSE(arcs.satisfied(1));
        EXPECT_THROW(arcs.setTarget(3),std::invalid_argument);
        EXPECT_THROW(mixed.setTarget(-1),std::invalid_argument);
    }
}

TEST(ArcTangency, SharedCoordinatesAccumulateDerivativesWithoutFalseTangency) {
    double px=2,py=1,cx=0,cy=0;
    ArcArcTangentError f({&px,&py,&cx,&cy,&px,&py,&cx,&cy},0);
    EXPECT_NEAR(f.evaluate(),-2*std::sqrt(5.),1e-12);
    EXPECT_FALSE(f.satisfied(1e-6));
    constexpr double h=1e-5;
    for (double* p:{&px,&py,&cx,&cy}) {
        const double old=*p;
        const auto gradient=f.gradient();
        *p=old+h;const double plus=f.evaluate(); const auto gp=f.gradient();
        *p=old-h;const double minus=f.evaluate(); const auto gm=f.gradient();
        *p=old;
        EXPECT_NEAR(gradient.at(p),(plus-minus)/(2*h),1e-8);
        for (double* q:{&px,&py,&cx,&cy})
            EXPECT_NEAR(f.secondDerivative(p,q),(gp.at(q)-gm.at(q))/(2*h),1e-8);
    }
    f.setTarget(1); EXPECT_FALSE(f.satisfied(1));
    f.setTarget(2); EXPECT_FALSE(f.satisfied(1));
}
