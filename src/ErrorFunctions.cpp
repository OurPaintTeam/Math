//
// Created by Eugene Bychkov on 12.11.2024.
//
#include "ErrorFunctions.h"

//------------------------- POINTSECDIST IMPLEMENTATION -------------------------
PointSectionDistanceError::PointSectionDistanceError(std::vector<Variable *> x, double error) : ErrorFunctions(x) {
    if (x.size() != 6) {
        throw std::invalid_argument("PointSectionDistanceError: wrong number of x");
    }
    Function* err = new Constant(error);
    Function* sq = new Constant(0.5);
    Function* pow2 = new Constant(2);
    Function* A = new Subtraction(x[5], x[3]);
    Function* B = new Subtraction(x[4], x[2]);
    Function* C = new Subtraction(new Multiplication(x[4], x[3]), new Multiplication(x[5], x[2]));
    Function* E = new Addition(new Subtraction(new Multiplication(A, x[0]), new Multiplication(B, x[1])), C);
    Function* F = new Division(E, new Power(new Addition(new Power(A, pow2), new Power(B, pow2)),sq));
    Function* G = new Subtraction(F, err);
    c_f = G;
}

PointOnSectionError::PointOnSectionError(std::vector<Variable *> x) : PointSectionDistanceError(x, 0){}

// ------------------------- POINTPOINTDIST IMPLEMENTATION -------------------------

PointPointDistanceError::PointPointDistanceError(std::vector<Variable *> x, double error) : ErrorFunctions(x) {
    if (x.size() != 4) {
        throw std::invalid_argument("PointPointDistanceError: wrong number of x");
    }
    Function *err = new Constant(error);
    Function *sq = new Constant(0.5);
    Function *pow2 = new Constant(2);
    Function *A = new Subtraction(x[2], x[0]);
    Function *B = new Subtraction(x[3], x[1]);
    Function *E = new Power(new Addition(new Power(A, pow2), new Power(B,pow2)),sq);
    Function *F = new Subtraction(E, err);
    c_f = F;
}
PointOnPointError::PointOnPointError(std::vector<Variable *> x) : PointPointDistanceError(x, 0){}

// ------------------------- SECSECPARALLEL IMPLEMENTATION -------------------------
SectionSectionParallelError::SectionSectionParallelError(std::vector<Variable *> x): ErrorFunctions(x) {
    if (x.size() != 8) {
        throw std::invalid_argument("SectionSectionParallelError: wrong number of x");
    }
    Function *v1 = new Subtraction(x[2], x[0]);
    Function *v2 = new Subtraction(x[3], x[1]);
    Function *w1 = new Subtraction(x[6], x[4]);
    Function *w2 = new Subtraction(x[7], x[5]);
    Function* F = new Subtraction(new Multiplication(v1, v2), new Multiplication(w1, w2));
    c_f = F;
}
//------------------------- SECSECPERPENDICULAR IMPLEMENTATION -------------------------
SectionSectionPerpendicularError::SectionSectionPerpendicularError(std::vector<Variable *> x): ErrorFunctions(x) {
    if (x.size() != 8) {
        throw std::invalid_argument("SectionSectionPerpendicularError: wrong number of x");
    }

    Function *v1 = new Subtraction(x[2], x[0]);
    Function *v2 = new Subtraction(x[3], x[1]);
    Function *w1 = new Subtraction(x[6], x[4]);
    Function *w2 = new Subtraction(x[7], x[5]);
    Function* F = new Addition(new Multiplication(v1, w1), new Multiplication(v2, w2));
    c_f = F;
}
//------------------------- SECTIONCIRCLEDISTANCE IMPLEMENTATION -------------------------
SectionCircleDistanceError::SectionCircleDistanceError(std::vector<Variable *> x, double error):ErrorFunctions(x) {
    if (x.size() != 7) {
        throw std::invalid_argument("SectionCircleDistanceError: wrong number of x");
    }
    // xs ys xe ye xc yc r
    Function *dist = new Addition(new Constant(error), new Constant(x[6]->evaluate()));
    Function *pow2 = new Constant(2);
    Function *A = new Subtraction(x[3], x[1]);
    Function *B = new Subtraction(x[2], x[0]);
    Function *C = new Subtraction(new Multiplication(x[2], x[1]), new Multiplication(x[0], x[3]));
    Function *e = new Division(new Addition(new Subtraction(A, B), C), new Sqrt(new Addition(new Power(A,A), new Power(B, B))));
    Function *F = new Subtraction(e, dist);
    c_f = F;
}

SectionOnCircleError::SectionOnCircleError(std::vector<Variable *> x) : SectionCircleDistanceError(x, 0) {}

//------------------------- SECTIONINCIRCLE IMPLEMENTATION -------------------------
SectionInCircleError::SectionInCircleError(std::vector<Variable *> x) : ErrorFunctions(x) {
    //No implementation on simple Functions without Max
}

//------------------------- SECTIONSECTIONANGLE IMPLEMENTATION -------------------------
SectionSectionAngleError::SectionSectionAngleError(std::vector<Variable *> x, double error):ErrorFunctions(x){
    if (x.size() != 8) {
        throw std::invalid_argument("SectionSectionAngleError: wrong number of x");
    }
    // x1s y1s x1e y1e x2s y2s x2e y2e
    Function *err = new Constant(error);
    Constant *PI = new Constant(M_PI);
    Constant *PI_2deg = new Constant(180);
    Function *pow2 = new Constant(2);
    Function *v1 =  new Subtraction(x[2], x[0]);
    Function *v2 =  new Subtraction(x[3], x[1]);
    Function *w1 =  new Subtraction(x[6], x[4]);
    Function *w2 =  new Subtraction(x[7], x[5]);
    Function *dot_product = new Addition(new Multiplication(v1, w1), new Multiplication(v2, w2));
    Function* mag_v = new Sqrt(new Addition(new Power(v1, pow2), new Power(v2, pow2)));
    Function* mag_w = new Sqrt(new Addition(new Power(w1, pow2), new Power(w2, pow2)));
    Function* cos_angle = new Division(dot_product, new Multiplication(mag_v, mag_w));
    Function* angle = new Multiplication(new Division(new Acos(cos_angle), PI), PI_2deg);
    c_f = new Subtraction(angle, err);
}

