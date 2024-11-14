//
// Created by Eugene Bychkov on 12.11.2024.
//
#include "gtest/gtest.h"
#include "ErrorFunctions.h"

//------------------------- POINTSECDIST TESTS -------------------------
TEST(PointSectionDistanceErrorTest, CorrectErrorValue) {
    double p1[] = {0.0}; double p2[] = {0.0};  // Point (1,1)
    double s1[] = {4.0}; double s2[] = {0.0};  // Segment start (4,0)
    double e1[] = {4.0}; double e2[] = {3.0};  // Segment end (4,3)

    std::vector<Variable*> variables = {
            new Variable(p1), new Variable(p2),
            new Variable(s1), new Variable(s2),
            new Variable(e1), new Variable(e2)
    };

    PointSectionDistanceError errorFunc(variables, 10);
    EXPECT_NEAR(std::abs(errorFunc.evaluate()), 14, 1e-2);
}

TEST(PointSectionDistanceErrorTest, IncorrectVariableCount) {
    double p1[] = {1.0}; double p2[] = {1.0}; double s1[] = {0.0};
    std::vector<Variable*> incorrectVariables = { new Variable(p1), new Variable(p2), new Variable(s1) };
    EXPECT_THROW(PointSectionDistanceError errorFunc(incorrectVariables, 1.0), std::invalid_argument);
}

//------------------------- POINTONSEC TESTS -------------------------
TEST(PointOnSectionErrorTest, CorrectZeroErrorValue) {
    double p1[] = {2.0}; double p2[] = {1.5};  // Point on segment
    double s1[] = {0.0}; double s2[] = {0.0};  // Segment start (0,0)
    double e1[] = {4.0}; double e2[] = {3.0};  // Segment end (4,3)

    std::vector<Variable*> variables = {
            new Variable(p1), new Variable(p2),
            new Variable(s1), new Variable(s2),
            new Variable(e1), new Variable(e2)
    };

    PointOnSectionError errorFunc(variables);
    EXPECT_NEAR(errorFunc.evaluate(), 0.0, 1e-5);
}

//------------------------- POINTPOINTDIST TESTS -------------------------
TEST(PointPointDistanceErrorTest, CorrectErrorValue) {
    double p1[] = {1.0}; double p2[] = {1.0}; double p3[] = {4.0}; double p4[] = {5.0};

    std::vector<Variable*> variables = { new Variable(p1), new Variable(p2), new Variable(p3), new Variable(p4) };
    PointPointDistanceError errorFunc(variables, 5.0);
    EXPECT_NEAR(errorFunc.evaluate(), 0.0, 1e-5);
}

TEST(PointPointDistanceErrorTest, IncorrectVariableCount) {
    double p1[] = {1.0}; double p2[] = {1.0};
    std::vector<Variable*> incorrectVariables = { new Variable(p1), new Variable(p2) };
    EXPECT_THROW(PointPointDistanceError errorFunc(incorrectVariables, 5.0), std::invalid_argument);
}

//------------------------- POINTONPOINT TESTS -------------------------
TEST(PointOnPointErrorTest, CorrectZeroErrorValue) {
    double p1[] = {2.0}; double p2[] = {3.0}; double p3[] = {2.0}; double p4[] = {3.0};

    std::vector<Variable*> variables = { new Variable(p1), new Variable(p2), new Variable(p3), new Variable(p4) };
    PointOnPointError errorFunc(variables);
    EXPECT_NEAR(errorFunc.evaluate(), 0.0, 1e-5);
}

//------------------------- SECSECPARALLEL TESTS -------------------------
TEST(SectionSectionParallelErrorTest, CorrectParallelErrorValue) {
    double p1[] = {1.0}; double p2[] = {1.0}; double p3[] = {4.0}; double p4[] = {4.0};
    double s1[] = {2.0}; double s2[] = {2.0}; double s3[] = {5.0}; double s4[] = {5.0};

    std::vector<Variable*> variables = {
            new Variable(p1), new Variable(p2), new Variable(p3), new Variable(p4),
            new Variable(s1), new Variable(s2), new Variable(s3), new Variable(s4)
    };

    SectionSectionParallelError errorFunc(variables);
    EXPECT_NEAR(errorFunc.evaluate(), 0.0, 1e-5);
}

TEST(SectionSectionParallelErrorTest, IncorrectVariableCount) {
    double p1[] = {1.0}; double p2[] = {1.0}; double p3[] = {4.0}; double p4[] = {4.0};
    std::vector<Variable*> incorrectVariables = { new Variable(p1), new Variable(p2), new Variable(p3), new Variable(p4) };
    EXPECT_THROW(SectionSectionParallelError errorFunc(incorrectVariables), std::invalid_argument);
}

//------------------------- SECSECPERPENDICULAR TESTS -------------------------
TEST(SectionSectionPerpendicularErrorTest, CorrectPerpendicularErrorValue) {
    double p1[] = {0.0}; double p2[] = {0.0}; double p3[] = {1.0}; double p4[] = {1.0};
    double s1[] = {0.0}; double s2[] = {0.0}; double s3[] = {-1.0}; double s4[] = {1.0};

    std::vector<Variable*> variables = {
            new Variable(p1), new Variable(p2), new Variable(p3), new Variable(p4),
            new Variable(s1), new Variable(s2), new Variable(s3), new Variable(s4)
    };

    SectionSectionPerpendicularError errorFunc(variables);
    EXPECT_NEAR(errorFunc.evaluate(), 0.0, 1e-5);
}

TEST(SectionSectionPerpendicularErrorTest, IncorrectVariableCount) {
    double p1[] = {0.0}; double p2[] = {0.0}; double p3[] = {1.0};
    std::vector<Variable*> incorrectVariables = { new Variable(p1), new Variable(p2), new Variable(p3) };
    EXPECT_THROW(SectionSectionPerpendicularError errorFunc(incorrectVariables), std::invalid_argument);
}
//------------------------- SECTIONCIRCLEDISTANCE TESTS -------------------------
TEST(SectionCircleDistanceErrorTest, CorrectErrorValue) {
    double xs[] = {0.0}; double ys[] = {0.0};
    double xe[] = {4.0}; double ye[] = {0.0};
    double xc[] = {2.0}; double yc[] = {2.0};
    double r[]  = {2.0};

    std::vector<Variable*> variables = {
            new Variable(xs), new Variable(ys),
            new Variable(xe), new Variable(ye),
            new Variable(xc), new Variable(yc),
            new Variable(r)
    };

    SectionCircleDistanceError errorFunc(variables, 0.0);
    EXPECT_NEAR(errorFunc.evaluate(), -2.2495131, 1e-5);
}

TEST(SectionCircleDistanceErrorTest, IncorrectVariableCount) {
    double xs[] = {0.0}; double ys[] = {0.0};
    double xe[] = {4.0}; double ye[] = {0.0};
    double xc[] = {2.0}; double yc[] = {2.0};

    std::vector<Variable*> variables = {
            new Variable(xs), new Variable(ys),
            new Variable(xe), new Variable(ye),
            new Variable(xc), new Variable(yc)
    };

    EXPECT_THROW(SectionCircleDistanceError errorFunc(variables, 0.0), std::invalid_argument);
}

//------------------------- SECTIONONCIRCLE TESTS -------------------------
TEST(SectionOnCircleErrorTest, CorrectZeroErrorValue) {
    double xs[] = {2.0}; double ys[] = {0.0};
    double xe[] = {0.0}; double ye[] = {2.0};
    double xc[] = {0.0}; double yc[] = {0.0};
    double r[]  = {2.0};

    std::vector<Variable*> variables = {
            new Variable(xs), new Variable(ys),
            new Variable(xe), new Variable(ye),
            new Variable(xc), new Variable(yc),
            new Variable(r)
    };

    SectionOnCircleError errorFunc(variables);
    EXPECT_NEAR(errorFunc.evaluate(), -2.0, 1e-5);
}

TEST(SectionOnCircleErrorTest, IncorrectVariableCount) {
    double xs[] = {2.0}; double ys[] = {0.0};
    double xe[] = {0.0}; double ye[] = {2.0};
    double xc[] = {0.0}; double yc[] = {0.0};

    std::vector<Variable*> variables = {
            new Variable(xs), new Variable(ys),
            new Variable(xe), new Variable(ye),
            new Variable(xc), new Variable(yc)
    };

    EXPECT_THROW(SectionOnCircleError errorFunc(variables), std::invalid_argument);
}
//------------------------- SECTIONSECTIONANGLE TESTS -------------------------
TEST(SectionSectionAngleErrorTest, CorrectAngleErrorValue) {
    double x1s[] = {0.0}; double y1s[] = {0.0};
    double x1e[] = {1.0}; double y1e[] = {0.0};
    double x2s[] = {2.0}; double y2s[] = {2.0};
    double x2e[] = {3.0}; double y2e[] = {3.0};

    std::vector<Variable*> variables = {
            new Variable(x1s), new Variable(y1s),
            new Variable(x1e), new Variable(y1e),
            new Variable(x2s), new Variable(y2s),
            new Variable(x2e), new Variable(y2e)
    };

    SectionSectionAngleError* errorFunc = new SectionSectionAngleError(variables, 45);
    EXPECT_NEAR(errorFunc->evaluate(), 0.0, 1e-2);
}

TEST(SectionSectionAngleErrorTest, IncorrectVariableCount) {
    double x1s[] = {0.0}; double y1s[] = {0.0};
    double x1e[] = {1.0}; double y1e[] = {0.0};
    double x2s[] = {0.0}; double y2s[] = {0.0};

    std::vector<Variable*> variables = {
            new Variable(x1s), new Variable(y1s),
            new Variable(x1e), new Variable(y1e),
            new Variable(x2s), new Variable(y2s)
    };

    EXPECT_THROW(SectionSectionAngleError errorFunc(variables, 90.0), std::invalid_argument);
}