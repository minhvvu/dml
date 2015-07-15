/*
 * EMKMeans.h
 *
 *  Created on: Jun 4, 2015
 *      Author: vvminh
 *
 * Interface for all KMeans algorithm
 */

#ifndef EMKMEANS_EMKMEANS_H_
#define EMKMEANS_EMKMEANS_H_

#include <memory>
#include <vector>

#include "EMResult.h"
#include "../gaussian/Gaussian.h"
#include "../utils/Eigen3.h"

namespace dml {

enum CovType {
	COV_NONE, COV_DIAG, COV_FULL
};

typedef std::shared_ptr<Gaussian> GaussianPtr;

class EMKMeans {
public:
	EMKMeans(const Eigen::Ref<const Eigen::MatrixXf>& dataset,
		const int numClts, const CovType type = COV_NONE);
	virtual ~EMKMeans() {}

	std::vector<int> doClustering(const int maxIteration = 100, const float minObjFuncChange = 0.01f);
	virtual EMResult getResult();
    float getCurrentCost();

protected:

	void createMixtures();
	void runEM();
	void runEStep(const std::vector<int>& randomIndex);
	void adaptMixturesSize();
	void assignData();
	void runMStep();
	bool checkConvergence(const float currentCost);

	virtual void createInitCenters();
	virtual void doVeryFirstClustering();
	virtual void findBestCluster(const std::vector<int>& randomIndex) = 0;
	virtual void updateMixtures() = 0;
	virtual float calculateObjFunc() = 0;

protected:
	CovType covType;
	int nData = 0;				// number of data point in the data set
	int nDims = 0;				// number of dimension (features) of each data point
	int nClusters = 0;			// the number of cluster

	Eigen::MatrixXf data; 		// one column is one data point
	std::vector<int> vAssign;	// the assignment of each data point to its cluster
	                            // vAssigment[data_point_id] => cluster_id
	std::vector<GaussianPtr> vMixture;	// the mixture of Gaussians
	std::vector<float> vObjFuncCached;	// all value of obj function at each iteration

	int maxIter = 0; 			// maximum iterator, if exceed the maxIter, then convergence!
	int currIter = 0; 			// current iterator
	float minChange = 0.0f;		// the minimun change of objetive function
	                            // if the change between 2 iterator is smaller than the minChange, then convergence!
};

} /* namespace dml */

#endif /* EMKMEANS_EMKMEANS_H_ */
