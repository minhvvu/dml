/*
 * DiagGaussian.cpp
 *
 *  Created on: Jun 22, 2015
 *      Author: vvminh
 */

#ifndef GAUSSIAN_DIAGGAUSSIAN_
#define GAUSSIAN_DIAGGAUSSIAN_

#include "Gaussian.h"

#include <map>
#include <utility>
#include <vector>
#include <iostream>
 
namespace dml {

using namespace Eigen;

class DiagGaussian : public Gaussian {
public:
	DiagGaussian(int id, int size, int dimensions)
		: Gaussian(id, size, dimensions) {
		// when covDiag is not calculated, it is just simple Identity mat
		// so we the distance that uses covDiad is calculated like Euclidean
		covDiag = VectorXf::Ones(dimensions);
	}

	virtual ~DiagGaussian(){}

	virtual void updateConstraintImpact(const Ref<const MatrixXf>& X, 
		const std::vector<int>& vAssign, const ConstraintPtr constraints) {

		VectorXf mlImpact = getMLImpact(X, vAssign, constraints->ML);
		VectorXf clImpact = getCLImpact(X, vAssign, constraints->CL);
		updateCovDiag(mlImpact, clImpact, constraints->mlConst, constraints->clConst);
		calculateLogDet();

		// std::cout << "\tupdate clt " << cltId << ": maxDist = " << maxDist
		// 	<< "\tfarthest pair: (" << farthest1 << ", " << farthest2 << ")"
		// 	<< "\tlogDet = " << logDet << '\n';
	}		

protected:

	VectorXf getMLImpact(const Ref<const MatrixXf>& X, 
		const std::vector<int>& vAssign, const ConstraintMap& ML) {
		
		int numViolation = 0;
		VectorXf impact = VectorXf::Zero(nDims);
		for (const auto& it : ML) {
			const int idx1 = it.first;
			if ( (GLOBAL_GAUSSIAN_ID == this->cltId) || (vAssign[idx1] == this->cltId) ){
				for (const int& idx2 : it.second) {
					if (vAssign[idx1] != vAssign[idx2]) {
						numViolation ++;
						impact += (X.col(idx1) - X.col(idx2)).array().square().matrix();
					}
				}
			}
		}
		return 0.5 * impact;
	}

	VectorXf getCLImpact(const Ref<const MatrixXf>& X, 
		const std::vector<int>& vAssign, const ConstraintMap& CL) {

		VectorXf impact = VectorXf::Zero(nDims);
		int numViolation = 0;
		for (const auto& it : CL) {
			const int idx1 = it.first;
			if ( (GLOBAL_GAUSSIAN_ID == this->cltId) || (vAssign[idx1] == this->cltId) ){
				for (const int& idx2 : it.second) {
					if (vAssign[idx1] == vAssign[idx2]) {
						numViolation ++;
						impact -= (X.col(idx1) - X.col(idx2)).array().square().matrix();
					}
				}
			}
		}

		if (numViolation > 0) {
			impact += numViolation * (X.col(farthest1) - X.col(farthest2)).array().square().matrix();
		}
		return impact;
	}

	void updateCovDiag(const Ref<const VectorXf>& mlImpact, const Ref<const VectorXf>& clImpact,
		const float mlConst, const float clConst) {

		covDiag = (data.colwise() - mean).array().square().matrix().rowwise().sum();
		covDiag += mlConst * mlImpact;
		covDiag += clConst * clImpact;
		covDiag /= (1.0f * nSize);
		covDiag.array() += epsilon;
	}

	void calculateLogDet() {
		logDet = -covDiag.array().abs().log().sum() / nSize;
	}

	virtual float applyDistance(const ConstVectorRef& v1, const ConstVectorRef& v2) {
		float dist = ( 
			(v1 - v2).array().square() / 
			covDiag.array()
		).sum();

		return (dist > 0.0f) ? std::sqrt(dist) : 0.0f;
	}

protected:
	VectorXf covDiag;
	float epsilon = 0.001f;

};

} /* namespace dml */

#endif /* GAUSSIAN_DIAGGAUSSIAN_ */	