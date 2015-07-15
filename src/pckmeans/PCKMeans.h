/*
 * PCKMeans.h
 *
 *  Created on: Jun 22, 2015
 *      Author: vvminh
 */

#ifndef PCKMEANS_PCKMEANS_H_
#define PCKMEANS_PCKMEANS_H_

#include "../utils/Eigen3.h"
#include "../emkmeans/EMKMeans.h"
#include "../constraint/ConstraintsManager.h"

#include <string>

namespace dml {

class PCKMeans : public EMKMeans {
public:
	PCKMeans(const ConstMatrixRef& dataset, const int numClts, 
		const std::string constraintFileName, const CovType type = COV_NONE);
		
	virtual ~PCKMeans();

	virtual void createInitCenters();
	virtual void doVeryFirstClustering();
	virtual void findBestCluster(const std::vector<int>& randomIndex);

	float getVariance(const int idx, const int cltId);
	float getMustLinksPenalty(const int idx1, const int cltId1);
	float getCannotLinksPenalty(const int idx1, const int cltId1);

	virtual void updateMixtures();
	virtual float calculateObjFunc();
	virtual EMResult getResult();
	
protected:
	float totalVariance();
	float totalMLPenalty();
	float totalCLPenalty();

	// using the same codebase for global metric and local metric
	virtual float distanceByCluster(int idx1, int idx2, int cltId = -1);
	virtual float distanceToMeanOfCluster(int idx, int cltId = -1);
	virtual float maxDistanceByCluster(int cltId = -1);
	virtual float logDetByCluster(int cltId = -1);

protected:
	ConstraintPtr constr;
	
public:
	int countMLViolation = 0;
	int countCLViolation = 0;
};

} /* namespace dml */

#endif /* PCKMEANS_PCKMEANS_H_ */	
