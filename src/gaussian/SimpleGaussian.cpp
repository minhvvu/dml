/*
 * SimpleGaussian.cpp
 *
 *  Created on: Jun 22, 2015
 *      Author: vvminh
 */

#ifndef GAUSSIAN_SIMPLEGAUSSIAN_
#define GAUSSIAN_SIMPLEGAUSSIAN_

#include "Gaussian.h"

namespace dml {

class SimpleGaussian : public Gaussian {
public:
	SimpleGaussian(int id, int size, int dimensions)
		: Gaussian(id, size, dimensions) {}

	virtual ~SimpleGaussian(){}
	
	virtual void updateConstraintImpact(const ConstMatrixRef& X, 
		const std::vector<int>& vAssign, const ConstraintPtr constraints) {}

	virtual float applyDistance(const ConstVectorRef& v1, const ConstVectorRef& v2) {
		return (v1 - v2).norm();
	}

};

} /* namespace dml */

#endif /* GAUSSIAN_SIMPLEGAUSSIAN_ */	