/*
 * Gaussian.h
 *
 *  Created on: Jun 19, 2015
 *      Author: vvminh
 */

#ifndef GAUSSIAN_GAUSSIAN_H_
#define GAUSSIAN_GAUSSIAN_H_

#include "../utils/Eigen3.h"
#include "../constraint/ConstraintsManager.h"
#include <vector>

namespace dml {

typedef Eigen::Ref<const Eigen::VectorXf> ConstVectorRef;
typedef Eigen::Ref<const Eigen::MatrixXf> ConstMatrixRef;
const static int GLOBAL_GAUSSIAN_ID = -1;

class Gaussian {
public:
	Gaussian(const int id, const int size, const int dimensions);
	virtual ~Gaussian(){}

	int getClusterId();
	float getMaxDistance();
	float getLogDet();
	const Eigen::VectorXf getMean();
	void setInitCenter(const ConstVectorRef& initCenter);

	void adaptNewSize(const int size);
	void insertDataPoint(const ConstVectorRef& dp);

	void setData(const ConstMatrixRef& X);

	void updateMean();
	virtual void updateConstraintImpact(const ConstMatrixRef& X, 
		const std::vector<int>& vAssign, const ConstraintPtr constraints) = 0;
	virtual void cacheDistPoint2Point(const ConstMatrixRef& X);
	virtual void cacheDistPoint2Mean(const ConstMatrixRef& X);

	float distance(const int idx1, const int idx2);
	float distanceToMean(const int idx);

	void debugCachedDistance();
	
	// note euclidean dis is special case when DIAG_COV = I
	virtual float applyDistance(const ConstVectorRef& v1, const ConstVectorRef& v2) = 0;

protected:

	int cltId = 0;
	int nAssigned = 0;
	int nSize = 0;
	int nDims = 0;
	float maxDist = 0.0f;
	float logDet = 0.0f;
	
	Eigen::MatrixXf data;
	Eigen::VectorXf mean;

	Eigen::MatrixXf distP2P;
	Eigen::VectorXf distP2M;
	int farthest1 = 0;
	int farthest2 = 0;
};

} /* namespace dml */

#endif /* GAUSSIAN_GAUSSIAN_H_ */
