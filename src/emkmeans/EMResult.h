/*
 * EMResult.h
 *
 *  Created on: Jul 7, 2015
 *      Author: vvminh
 *
 * data object to save the result of EMKMeans algo
 * the child class of EMKMeans will add infos to this object
 */

#ifndef EMKMEANS_EMRESULT_H_
#define EMKMEANS_EMRESULT_H_

#include <string>

namespace dml {

/**
 * class to store result of EM algo, for each experimentation,
 * or for accumulate average result of many experimentations.
 */
class EMResult {
public:
    float iterTerminate = 0.0f;        // number of iteration before convergence
    float cost = 0.0f;                 // cost function at convergence
    float reachLocalMinimal = 0.0f;    // convergence at local minimal or not
    float nConstraintOriginal = 0.0f;  // number of constraint original
    float nConstraintDeduced = 0.0f;   // number of constraint deduced
    float nMLStart = 0.0f;             // number of MustLink when starting EM
    float nCLStart = 0.0f;             // number of CannotLink when starting EM
    float nMLViolation = 0.0f;         // number of ML violation when EM terminates
    float nCLViolation = 0.0f;         // number of CM violation when EM terminates
    float mlConst = 0.0f;              // mustlink coeff constant used in cost func
    float clConst = 0.0f;              // cannotlink coeff constant used
    float vMeasure = 0.0f;             // measure performance with ground truth
    float duration = 0.0f;             // running time in millisecond

    void add(const EMResult& r) {
        this->iterTerminate +=      r.iterTerminate;
        this->cost +=               r.cost;
        this->reachLocalMinimal +=  r.reachLocalMinimal;
        this->nConstraintOriginal += r.nConstraintOriginal;
        this->nConstraintDeduced += r.nConstraintDeduced;
        this->nMLStart +=           r.nMLStart;
        this->nCLStart +=           r.nCLStart;
        this->nMLViolation +=       r.nMLViolation;
        this->nCLViolation +=       r.nCLViolation;
        this->vMeasure +=           r.vMeasure;
        this->duration +=           r.duration;
    }

    void divise(const float factor) {
        this->iterTerminate         /= factor;
        this->cost                  /= factor;
        this->reachLocalMinimal     /= factor;
        this->nConstraintOriginal   /= factor;
        this->nConstraintDeduced    /= factor;
        this->nMLStart              /= factor;
        this->nCLStart              /= factor;
        this->nMLViolation          /= factor;
        this->nCLViolation          /= factor;
        this->vMeasure              /= factor;
        this->duration              /= factor;
    }

    std::string toJson() const {
        std::string json = "";
        json += "{\n";
        json += ("\t\"iterTerminate\":\t" +       std::to_string(iterTerminate)        + ",\n");
        json += ("\t\"cost\":\t" +                std::to_string(cost)                 + ",\n");
        json += ("\t\"reachLocalMinimal\":\t" +   std::to_string(reachLocalMinimal)    + ",\n");
        json += ("\t\"nConstraintOriginal\":\t" + std::to_string(nConstraintOriginal)  + ",\n");
        json += ("\t\"nConstraintDeduced\":\t" +  std::to_string(nConstraintDeduced)   + ",\n");
        json += ("\t\"nMLStart\":\t" +            std::to_string(nMLStart)             + ",\n");
        json += ("\t\"nCLStart\":\t" +            std::to_string(nCLStart)             + ",\n");
        json += ("\t\"nMLViolation\":\t" +        std::to_string(nMLViolation)         + ",\n");
        json += ("\t\"nCLViolation\":\t" +        std::to_string(nCLViolation)         + ",\n");
        json += ("\t\"vMeasure\":\t" +            std::to_string(vMeasure)             + ",\n");
        json += ("\t\"duration\":\t" +            std::to_string(duration)             + ",\n");
        json += ("\t\"mlConst\":\t" +             std::to_string(mlConst)              + ",\n");
        json += ("\t\"clConst\":\t" +             std::to_string(clConst)              + "\n");
        json += "}";
        return json;
    }
};

} /* namespace dml */

#endif /* EMKMEANS_EMRESULT_H_ */
