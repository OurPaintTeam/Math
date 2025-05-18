#include "ErrorFunction.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

//------------------------- POINTSECDIST IMPLEMENTATION -------------------------
PointSectionDistanceError::PointSectionDistanceError(std::vector<Variable* > x, double error) : ErrorFunction(x) {
    v_error = error;
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
    Function* F = new Division(E, new Power(new Addition(new Power(A->clone(), pow2), new Power(B->clone(), pow2->clone())),sq));
    Function* G = new Subtraction(F, err);
    c_f = G;
}
Function *PointSectionDistanceError::clone() const {
    return new PointSectionDistanceError(m_X, v_error);
}

// ------------------------- POINTONSECTION IMPLEMENTATION -------------------------
PointOnSectionError::PointOnSectionError([[maybe_unused]] std::vector<Variable* > x) : PointSectionDistanceError(x, 0){}
Function *PointOnSectionError::clone() const {
    return new PointOnSectionError(m_X);
}

// ------------------------- POINTPOINTDIST IMPLEMENTATION -------------------------

PointPointDistanceError::PointPointDistanceError(std::vector<Variable* > x, double error) : ErrorFunction(x) {
    if (x.size() != 4) {
        throw std::invalid_argument("PointPointDistanceError: wrong number of x");
    }
    v_error = error;
    Function* dx = new Subtraction(x[2], x[0]);  // x2 - x1
    Function* dy = new Subtraction(x[3], x[1]);  // y2 - y1
    Function* dx2 = new Multiplication(dx, dx->clone());  // (x2 - x1)(x2 - x1)
    Function* dy2 = new Multiplication(dy, dy->clone());  // (y2 - y1)(y2 - y1)
    Function* dist2 = new Addition(dx2, dy2);    // (x2 - x1)(x2 - x1) + (y2 - y1)(y2 - y1)
    Function* target = new Constant(error * error);
    c_f = new Subtraction(dist2, target);
}
Function *PointPointDistanceError::clone() const {
    return new PointPointDistanceError(m_X, v_error);
}

// ------------------------- POINTONPOINT IMPLEMENTATION -------------------------
PointOnPointError::PointOnPointError([[maybe_unused]] std::vector<Variable *> x) : PointPointDistanceError(x, 0){}
Function *PointOnPointError::clone() const {
    return new PointOnPointError(m_X);
}

// ------------------------- SECSECPARALLEL IMPLEMENTATION -------------------------
SectionSectionParallelError::SectionSectionParallelError(std::vector<Variable* > x): ErrorFunction(x) {
    if (x.size() != 8) {
        throw std::invalid_argument("SectionSectionParallelError: wrong number of x");
    }
    Function* F = new Subtraction(
            new Multiplication(
                    new Subtraction(x[2], x[0]),
                    new Subtraction(x[7], x[5])
            ),
            new Multiplication(
                    new Subtraction(x[3], x[1]),
                    new Subtraction(x[6], x[4])
            )
    );

    c_f = F;
}
Function *SectionSectionParallelError::clone() const {
    return new SectionSectionParallelError(m_X);
}

//------------------------- SECSECPERPENDICULAR IMPLEMENTATION -------------------------
SectionSectionPerpendicularError::SectionSectionPerpendicularError(std::vector<Variable* > x): ErrorFunction(x) {
    if (x.size() != 8) {
        throw std::invalid_argument("SectionSectionPerpendicularError: wrong number of x");
    }

    Function* F = new Addition(
            new Multiplication(
                    new Subtraction(x[2], x[0]),
                    new Subtraction(x[6], x[4])
            ),
            new Multiplication(
                    new Subtraction(x[3], x[1]),
                    new Subtraction(x[7], x[5])
            )
    );
    c_f = F;
}
Function *SectionSectionPerpendicularError::clone() const {
    return new SectionSectionPerpendicularError(m_X);
}

//------------------------- SECTIONCIRCLEDISTANCE IMPLEMENTATION -------------------------
SectionCircleDistanceError::SectionCircleDistanceError(std::vector<Variable* > x, double error):ErrorFunction(x) {
    if (x.size() != 7) {
        throw std::invalid_argument("SectionCircleDistanceError: wrong number of x");
    }
    // xs ys xe ye xc yc r
    v_error = error;
    Function *dist = new Addition(new Constant(error), new Constant(x[6]->evaluate()));
    Function *A = new Subtraction(x[3], x[1]);
    Function *B = new Subtraction(x[2], x[0]);
    Function *C = new Subtraction(new Multiplication(x[2], x[1]), new Multiplication(x[0], x[3]));
    Function *e = new Division(new Addition(new Subtraction(A, B), C), new Sqrt(new Addition(new Power(A->clone(),A->clone()), new Power(B->clone(), B->clone()))));
    Function *F = new Subtraction(e, dist);
    c_f = F;
}
Function *SectionCircleDistanceError::clone() const {
    return new SectionCircleDistanceError(m_X, v_error);
}

// ------------------------- SECTIONONCIRCLE IMPLEMENTATION -------------------------
SectionOnCircleError::SectionOnCircleError([[maybe_unused]] std::vector<Variable *> x) : SectionCircleDistanceError(x, 0) {}
Function *SectionOnCircleError::clone() const {
    return new SectionOnCircleError(m_X);
}

//------------------------- SECTIONINCIRCLE IMPLEMENTATION -------------------------
SectionInCircleError::SectionInCircleError([[maybe_unused]] std::vector<Variable *> x) : ErrorFunction(x) {
    //No implementation on simple Functions without Max
}
Function *SectionInCircleError::clone() const {
    return new SectionInCircleError(m_X);
}

//------------------------- SECTIONSECTIONANGLE IMPLEMENTATION -------------------------
SectionSectionAngleError::SectionSectionAngleError(std::vector<Variable *> x, double error):ErrorFunction(x){
    if (x.size() != 8) {
        throw std::invalid_argument("SectionSectionAngleError: wrong number of x");
    }
    // x1s y1s x1e y1e x2s y2s x2e y2e
    v_error = error;
    Function *err = new Constant(error);
    Constant *PI = new Constant(M_PI);
    Constant *PI_2deg = new Constant(180);
    Function *pow2 = new Constant(2);
    Function *v1 =  new Subtraction(x[2], x[0]);
    Function *v2 =  new Subtraction(x[3], x[1]);
    Function *w1 =  new Subtraction(x[6], x[4]);
    Function *w2 =  new Subtraction(x[7], x[5]);
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

ArcCenterOnPerpendicularError::ArcCenterOnPerpendicularError(std::vector<Variable *> x) : ErrorFunction(x) {
    if (x.size() != 6) {
        throw std::invalid_argument("ArcCenterOnPerpendicularError: wrong number of x");
    }
    // Вычисляем середину отрезка p1p2
    Function* midX = new Division(new Addition(x[0], x[2]), new Constant(2));
    Function* midY = new Division(new Addition(x[1], x[3]), new Constant(2));
    // Вычисляем вектор от середины к p3
    Function* vecX = new Subtraction(x[4], midX);
    Function* vecY = new Subtraction(x[5], midY);
    // Вычисляем вектор от p1 к p2
    Function* segX = new Subtraction(x[2], x[0]);
    Function* segY = new Subtraction(x[3], x[1]);
    // Проверяем, что векторы перпендикулярны (скалярное произведение равно 0)
    Function* dotProduct = new Addition(new Multiplication(vecX, segX), new Multiplication(vecY, segY));
    c_f = dotProduct;
}

Function* ArcCenterOnPerpendicularError::clone() const {
    return new ArcCenterOnPerpendicularError(m_X);
}
