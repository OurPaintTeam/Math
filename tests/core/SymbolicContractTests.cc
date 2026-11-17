#include "Function.h"
#include <gtest/gtest.h>
#include <memory>

TEST(SymbolicContract, VariablePowerAndLogBaseDerivativesRemainLive) {
    double a=2,b=3;
    Variable x(&a),y(&b);
    Power power(x.clone(),y.clone());
    std::unique_ptr<Function> dx(power.derivative(&x)),dy(power.derivative(&y));
    EXPECT_NEAR(dx->evaluate(),12,1e-12);
    EXPECT_NEAR(dy->evaluate(),8*std::log(2.0),1e-12);
    b=4;
    EXPECT_NEAR(dx->evaluate(),32,1e-12);
    Log log(x.clone(),y.clone());
    std::unique_ptr<Function> db(log.derivative(&x));
    EXPECT_NEAR(db->evaluate(),-std::log(4.0)/(2*std::log(2.0)*std::log(2.0)),1e-12);
    a=3;
    EXPECT_NEAR(db->evaluate(),-std::log(4.0)/(3*std::log(3.0)*std::log(3.0)),1e-12);
}
TEST(SymbolicContract, MinMaxSwitchAfterDifferentiationAndModuloSimplifiesSafely) {
    double a=1,b=2;
    Variable x(&a),y(&b);
    Max maximum(x.clone(),y.clone());
    Min minimum(x.clone(),y.clone());
    std::unique_ptr<Function> maxD(maximum.derivative(&x)),minD(minimum.derivative(&x));
    EXPECT_EQ(maxD->evaluate(),0); EXPECT_EQ(minD->evaluate(),1);
    a=3;
    EXPECT_EQ(maxD->evaluate(),1); EXPECT_EQ(minD->evaluate(),0);
    Mod mod(x.clone(),y.clone());
    std::unique_ptr<Function> simplified(mod.simplify());
    EXPECT_EQ(simplified->evaluate(),1);
    EXPECT_THROW(mod.derivative(&x),std::logic_error);
    b=2.5;
    EXPECT_EQ(simplified->evaluate(),0.5);
}
