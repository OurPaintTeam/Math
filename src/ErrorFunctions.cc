#include "ErrorFunctions.h"

//------------------------- POINTSECDIST IMPLEMENTATION -------------------------
PointSectionDistanceError::PointSectionDistanceError(std::vector<Variable *> x, double error) : ErrorFunctions(x) {
    v_error = error;
    if (x.size() != 6) {
        throw std::invalid_argument("PointSectionDistanceError: wrong number of x");
    }
    Function* err = new Constant(error);
    Function* sq = new Constant(0.5);
    Function* pow2 = new Constant(2);
    Function* A = new Subtraction(x[5]->clone(), x[3]->clone());
    Function* B = new Subtraction(x[4]->clone(), x[2]->clone());
    Function* C = new Subtraction(new Multiplication(x[4]->clone(), x[3]->clone()), new Multiplication(x[5]->clone(), x[2]->clone()));
    Function* E = new Addition(new Subtraction(new Multiplication(A, x[0]->clone()), new Multiplication(B, x[1]->clone())), C);
    Function* F = new Division(E, new Power(new Addition(new Power(A->clone(), pow2), new Power(B->clone(), pow2->clone())),sq));
    Function* G = new Subtraction(F, err);
    c_f = G;
}
Function *PointSectionDistanceError::clone() const {
    return new PointSectionDistanceError(m_X, v_error);
}

// ------------------------- POINTONSECTION IMPLEMENTATION -------------------------
PointOnSectionError::PointOnSectionError(std::vector<Variable *> x) : PointSectionDistanceError(x, 0){}
Function *PointOnSectionError::clone() const {
    return new PointOnSectionError(m_X);
}

// ------------------------- POINTPOINTDIST IMPLEMENTATION -------------------------

PointPointDistanceError::PointPointDistanceError(std::vector<Variable *> x, double error) : ErrorFunctions(x) {
    if (x.size() != 4) {
        throw std::invalid_argument("PointPointDistanceError: wrong number of x");
    }
    v_error = error;
    Function *err = new Constant(error);
    Function *sq = new Constant(0.5);
    Function *pow2 = new Constant(2);
    Function *A = new Subtraction(x[2]->clone(), x[0]->clone());
    Function *B = new Subtraction(x[3]->clone(), x[1]->clone());
    Function *E = new Power(new Addition(new Power(A, pow2), new Power(B,pow2->clone())),sq);
    Function *F = new Subtraction(E, err);
    c_f = F;
}
Function *PointPointDistanceError::clone() const {
    return new PointPointDistanceError(m_X, v_error);
}

// ------------------------- POINTONPOINT IMPLEMENTATION -------------------------
PointOnPointError::PointOnPointError(std::vector<Variable *> x) : PointPointDistanceError(x, 0){}
Function *PointOnPointError::clone() const {
    return new PointOnPointError(m_X);
}

// ------------------------- SECSECPARALLEL IMPLEMENTATION -------------------------
SectionSectionParallelError::SectionSectionParallelError(std::vector<Variable *> x): ErrorFunctions(x) {
    if (x.size() != 8) {
        throw std::invalid_argument("SectionSectionParallelError: wrong number of x");
    }
    Function *v1 = new Subtraction(x[2]->clone(), x[0]->clone());
    Function *v2 = new Subtraction(x[3]->clone(), x[1]->clone());
    Function *w1 = new Subtraction(x[6]->clone(), x[4]->clone());
    Function *w2 = new Subtraction(x[7]->clone(), x[5]->clone());
    Function* F = new Subtraction(new Multiplication(v1, w2), new Multiplication(v2, w1));
    c_f = F;
}
Function *SectionSectionParallelError::clone() const {
    return new SectionSectionParallelError(m_X);
}

//------------------------- SECSECPERPENDICULAR IMPLEMENTATION -------------------------
SectionSectionPerpendicularError::SectionSectionPerpendicularError(std::vector<Variable *> x): ErrorFunctions(x) {
    if (x.size() != 8) {
        throw std::invalid_argument("SectionSectionPerpendicularError: wrong number of x");
    }

    Function *v1 = new Subtraction(x[2]->clone(), x[0]->clone());
    Function *v2 = new Subtraction(x[3]->clone(), x[1]->clone());
    Function *w1 = new Subtraction(x[6]->clone(), x[4]->clone());
    Function *w2 = new Subtraction(x[7]->clone(), x[5]->clone());
    Function* F = new Addition(new Multiplication(v1, w1), new Multiplication(v2, w2));
    c_f = F;
}
Function *SectionSectionPerpendicularError::clone() const {
    return new SectionSectionPerpendicularError(m_X);
}

//------------------------- SECTIONCIRCLEDISTANCE IMPLEMENTATION -------------------------
SectionCircleDistanceError::SectionCircleDistanceError(std::vector<Variable *> x, double error):ErrorFunctions(x) {
    if (x.size() != 7) {
        throw std::invalid_argument("SectionCircleDistanceError: wrong number of x");
    }
    // xs ys xe ye xc yc r
    v_error = error;
    Function *dist = new Addition(new Constant(error), new Constant(x[6]->evaluate()));
    Function *A = new Subtraction(x[3]->clone(), x[1]->clone());
    Function *B = new Subtraction(x[2]->clone(), x[0]->clone());
    Function *C = new Subtraction(new Multiplication(x[2]->clone(), x[1]->clone()), new Multiplication(x[0]->clone(), x[3]->clone()));
    Function *e = new Division(new Addition(new Subtraction(A, B), C), new Sqrt(new Addition(new Power(A->clone(),A->clone()), new Power(B->clone(), B->clone()))));
    Function *F = new Subtraction(e, dist);
    c_f = F;
}
Function *SectionCircleDistanceError::clone() const {
    return new SectionCircleDistanceError(m_X, v_error);
}

// ------------------------- SECTIONONCIRCLE IMPLEMENTATION -------------------------
SectionOnCircleError::SectionOnCircleError(std::vector<Variable *> x) : SectionCircleDistanceError(x, 0) {}
Function *SectionOnCircleError::clone() const {
    return new SectionOnCircleError(m_X);
}

//------------------------- SECTIONINCIRCLE IMPLEMENTATION -------------------------
SectionInCircleError::SectionInCircleError(std::vector<Variable *> x) : ErrorFunctions(x) {
    //No implementation on simple Functions without Max
}
Function *SectionInCircleError::clone() const {
    return new SectionInCircleError(m_X);
}

//------------------------- SECTIONSECTIONANGLE IMPLEMENTATION -------------------------
SectionSectionAngleError::SectionSectionAngleError(std::vector<Variable *> x, double error):ErrorFunctions(x){
    if (x.size() != 8) {
        throw std::invalid_argument("SectionSectionAngleError: wrong number of x");
    }
    // x1s y1s x1e y1e x2s y2s x2e y2e
    v_error = error;
    Function *err = new Constant(error);
    Constant *PI = new Constant(M_PI);
    Constant *PI_2deg = new Constant(180);
    Function *pow2 = new Constant(2);
    Function *v1 =  new Subtraction(x[2]->clone(), x[0]->clone());
    Function *v2 =  new Subtraction(x[3]->clone(), x[1]->clone());
    Function *w1 =  new Subtraction(x[6]->clone(), x[4]->clone());
    Function *w2 =  new Subtraction(x[7]->clone(), x[5]->clone());
    Function *dot_product = new Addition(new Multiplication(v1, w1), new Multiplication(v2, w2));
    Function* mag_v = new Sqrt(new Addition(new Power(v1->clone(), pow2), new Power(v2->clone(), pow2->clone())));
    Function* mag_w = new Sqrt(new Addition(new Power(w1->clone(), pow2->clone()), new Power(w2->clone(), pow2->clone())));
    Function* cos_angle = new Division(dot_product, new Multiplication(mag_v, mag_w));
    Function* angle = new Multiplication(new Division(new Acos(cos_angle), PI), PI_2deg);
    c_f = new Subtraction(angle, err);
}
Function *SectionSectionAngleError::clone() const {
    return new SectionSectionAngleError(m_X, v_error);
}
