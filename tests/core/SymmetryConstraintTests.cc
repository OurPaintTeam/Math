#include "ErrorFunction.h"
#include <gtest/gtest.h>
#include <limits>

TEST(SymmetryConstraint, CoincidentPointsAreValidOnAxisButCollapsedAxisIsUndefined) {
    double x=2,y=2,ax=0,ay=0,bx=4,by=4;
    SymmetryAlongError along({&x,&y,&x,&y,&ax,&ay,&bx,&by});
    SymmetryAcrossError across({&x,&y,&x,&y,&ax,&ay,&bx,&by});
    EXPECT_TRUE(along.satisfied(0));
    EXPECT_TRUE(across.satisfied(0));
    EXPECT_EQ(along.gradient().at(&x),0);
    EXPECT_NEAR(across.gradient().at(&x),-std::sqrt(0.5),1e-12);
    const auto gradient=across.gradient();
    constexpr double h=1e-5;
    x+=h; const double plus=across.evaluate(); x-=2*h;
    const double minus=across.evaluate(); x+=h;
    EXPECT_NEAR(gradient.at(&x),(plus-minus)/(2*h),1e-9);
    bx=ax; by=ay;
    EXPECT_FALSE(along.satisfied(1));
    EXPECT_FALSE(across.satisfied(1));
    across.setWeight(0);
    EXPECT_TRUE(across.satisfied(0));
    EXPECT_EQ(across.weightedValue(),0);
}

TEST(SymmetryConstraint, AxisVariantsShareLiveTargetsWeightsAndStableAverages) {
    double p=1e308,q=1e308;
    CoordinateAverageError average(std::vector<double*>{&p,&q},1e308);
    EXPECT_EQ(average.evaluate(),0);
    CoordinateDifferenceError same(std::vector<double*>{&p,&p});
    EXPECT_EQ(same.gradient().at(&p),0);
    CoordinateAverageError shared(std::vector<double*>{&p,&p},1e308);
    EXPECT_EQ(shared.gradient().at(&p),1);
    p=3; q=5; average.setTarget(-2); average.setWeight(3);
    EXPECT_EQ(average.evaluate(),6);
    EXPECT_EQ(average.weightedValue(),18);
    EXPECT_EQ(average.weightedGradient().at(&p),1.5);
    EXPECT_THROW(average.setTarget(std::numeric_limits<double>::infinity()),std::invalid_argument);
    EXPECT_THROW(same.setTarget(1),std::invalid_argument);
}
