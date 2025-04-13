#include "gtest/gtest.h"

#include "GradientOptimizer.h"
#include "TaskF.h"

TEST(OptimizerTest, SingleVariableQuadraticFunction) {
    //f(x) = (x - 3)^2
    double a = 0.0;
    Variable* x = new Variable(&a);
    Constant* c = new Constant(3);
    Constant* d = new Constant(2);
    Subtraction* e = new Subtraction(x->clone(), c);
    Function* f = new Power(e, d);
    std::vector<Variable*> variables = { x };
    TaskF* task = new TaskF(f, variables);
    GradientOptimizer optimizer(0.1, 1000); // learningRate=0.1, maxIterations=1000
    optimizer.setTask(task);

    optimizer.optimize();

    std::vector<double> result = optimizer.getResult();
    bool converged = optimizer.isConverged();
    double finalError = optimizer.getCurrentError();

    EXPECT_TRUE(converged);

    EXPECT_NEAR(result[0], 3.0, 1e-2);

    EXPECT_NEAR(finalError, 0.0, 1e-4);
    delete task;
    delete x;
}

TEST(OptimizerTest, MultiVariableQuadraticFunction) {
    // f(x, y) = (x - 2)^2 + (y + 5)^2
    double a = 1.0;
    double b = 2.0;
    Variable* x = new Variable(&a);
    Variable* y = new Variable(&b);
    Constant* c = new Constant(2.0);
    Constant* d = new Constant(5.0);
    Subtraction* e = new Subtraction(x->clone(), c);
    Addition* g = new Addition(y->clone(), d);
    Power* h = new Power(e, c->clone());
    Power* i = new Power(g, c->clone());
    Addition* f = new Addition(h, i);
    std::vector<Variable*> variables = { x, y };
    TaskF* task=  new TaskF(f, variables);

    GradientOptimizer optimizer(0.1, 1000); // learningRate=0.1, maxIterations=1000
    optimizer.setTask(task);

    optimizer.optimize();

    std::vector<double> result = optimizer.getResult();
    bool converged = optimizer.isConverged();
    double finalError = optimizer.getCurrentError();

    EXPECT_TRUE(converged);

    EXPECT_NEAR(result[0], 2.0, 1e-2);
    EXPECT_NEAR(result[1], -5.0, 1e-2);

    EXPECT_NEAR(finalError, 0.0, 1e-4);
    delete task;
    delete x;
    delete y;
}

TEST(OptimizerTest, DoesNotConvergeWithHighLearningRate) {
    double a = 0.0;
    Variable* x = new Variable(&a);
    Constant* b = new Constant(1.0);
    Constant* c = new Constant(2.0);
    Subtraction* d = new Subtraction(x->clone(), b);
    Power* f = new Power(d, c); // e = (x - 1)^2
    std::vector<Variable*> variables = { x };
    TaskF* task = new TaskF(f, variables);

    GradientOptimizer optimizer(10.0, 100); // learningRate=10.0, maxIterations=100
    optimizer.setTask(task);
    optimizer.optimize();
    std::vector<double> result = optimizer.getResult();
    bool converged = optimizer.isConverged();
    double finalError = optimizer.getCurrentError();

    EXPECT_FALSE(converged);
    EXPECT_GT(finalError, 1.0);

    delete task;
    delete x;
}

TEST(OptimizerTest, OptimizeWithoutSettingTask) {
    GradientOptimizer optimizer(0.1, 1000);

    optimizer.optimize();

    std::vector<double> result = optimizer.getResult();
    bool converged = optimizer.isConverged();
    double finalError = optimizer.getCurrentError();

    EXPECT_FALSE(converged);

    EXPECT_TRUE(result.empty());

    EXPECT_DOUBLE_EQ(finalError, 0.0);
}

TEST(OptimizerTest, AlreadyOptimal) {
    //f(x) = (x - 4)^2
    double a = 4.0;
    Variable* x = new Variable(&a);
    Constant* b = new Constant(4.0);
    Constant* c = new Constant(2.0);
    Subtraction* g = new Subtraction(x->clone(), b);
    Power* f = new Power(g, c);
    std::vector<Variable*> variables = { x };
    TaskF* task = new TaskF(f, variables);

    GradientOptimizer optimizer(0.1, 1000);
    optimizer.setTask(task);

    optimizer.optimize();

    std::vector<double> result = optimizer.getResult();
    bool converged = optimizer.isConverged();
    double finalError = optimizer.getCurrentError();

    EXPECT_TRUE(converged);

    EXPECT_DOUBLE_EQ(result[0], 4.0);

    EXPECT_DOUBLE_EQ(finalError, 0.0);

    delete task;
    delete x;
}