/*
 * GlobalMetricKMeans.h
 *
 *  Created on: Jun 26, 2015
 *      Author: vvminh
 */

#ifndef GLOBAL_METRIC_KMEANS_H_
#define GLOBAL_METRIC_KMEANS_H_

#include "../utils/Eigen3.h"
#include "../pckmeans/PCKMeans.h"
#include <string>

namespace dml {

enum DistanceType {
	DIST_EUCLIDEAN, DIST_MAHALANOBIS_DIAG, DIST_MAHALANOBIS_FULL
};

class GlobalMetricKMeans : public PCKMeans {
public:
	GlobalMetricKMeans(const ConstMatrixRef& dataset, const int numClts, 
		const std::string constraintFileName, const DistanceType type = DIST_EUCLIDEAN);
	virtual ~GlobalMetricKMeans();

	virtual void doVeryFirstClustering();
	virtual void updateMixtures();

protected:
	virtual float distanceByCluster(int idx1, int idx2, int cltId = -1);
	virtual float distanceToMeanOfCluster(int idx, int cltId = -1);
	virtual float maxDistanceByCluster(int cltId = -1);
	virtual float logDetByCluster(int cltId = -1);

private:
	void cacheGlobalDistPoint2Mean();

	Eigen::MatrixXf distP2M;

	GaussianPtr globalGaussian;
};

} /* namespace dml */

#endif /* GLOBAL_METRIC_KMEANS_H_ */	