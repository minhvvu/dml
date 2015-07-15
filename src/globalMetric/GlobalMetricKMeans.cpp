/*
 * GlobalMetricKMeans.cpp
 *
 *  Created on: Jun 27, 2015
 *      Author: vvminh
 */

#include "GlobalMetricKMeans.h"
#include "../utils/functionUtils.h"
#include "../gaussian/SimpleGaussian.cpp"
#include "../gaussian/DiagGaussian.cpp"
#include "../gaussian/FullGaussian.cpp"

#include <iostream>
#include <cassert>
#include <limits>

namespace dml {

using namespace Eigen;

GlobalMetricKMeans::GlobalMetricKMeans(const ConstMatrixRef& dataset, const int numClts,
	const std::string constraintFileName, const DistanceType distanceType)
	:PCKMeans(dataset, numClts, constraintFileName, COV_NONE) {
	distP2M = MatrixXf(nData, nClusters);

	switch (distanceType) {
		case DIST_EUCLIDEAN:
			globalGaussian = GaussianPtr( new SimpleGaussian(GLOBAL_GAUSSIAN_ID, nData, nDims) );
		break;
		case DIST_MAHALANOBIS_DIAG:
			globalGaussian = GaussianPtr( new DiagGaussian(GLOBAL_GAUSSIAN_ID, nData, nDims) );
		break;
		case DIST_MAHALANOBIS_FULL:
			globalGaussian = GaussianPtr( new FullGaussian(GLOBAL_GAUSSIAN_ID, nData, nDims) );
		break;
		default:
			assert(false && "Invalide distance type!");
		break;
	}
	globalGaussian->setData(dataset);
}

GlobalMetricKMeans::~GlobalMetricKMeans() {}

/*virtual*/ void GlobalMetricKMeans::doVeryFirstClustering() {
	// std::cout << "[trace]@ function : " <<  __PRETTY_FUNCTION__ << std::endl;
	globalGaussian->updateMean();
	globalGaussian->cacheDistPoint2Point(data);
	cacheGlobalDistPoint2Mean();
}

/*virtual*/ float GlobalMetricKMeans::distanceByCluster(int idx1, int idx2, int cltId) {
	return globalGaussian->distance(idx1, idx2);
}

/*virtual*/ float GlobalMetricKMeans::distanceToMeanOfCluster(int idx, int cltId) {
	return distP2M(idx, cltId);
}

/*virtual*/ float GlobalMetricKMeans::maxDistanceByCluster(int cltId) {
	return globalGaussian->getMaxDistance();
}

/*virtual*/ float GlobalMetricKMeans::logDetByCluster(int cltId) {
	return globalGaussian->getLogDet();
}

/*virtual*/ void GlobalMetricKMeans::updateMixtures() {
	for (int cltId = 0; cltId < nClusters; ++cltId) {
		vMixture.at(cltId)->updateMean();
	}
	globalGaussian->updateMean();
	globalGaussian->updateConstraintImpact(data, vAssign, constr);
	globalGaussian->cacheDistPoint2Point(data);
	cacheGlobalDistPoint2Mean();
}

void GlobalMetricKMeans::cacheGlobalDistPoint2Mean() {
	distP2M.setZero();
	for (int cltId = 0; cltId < nClusters; ++cltId) {
		VectorXf mean = vMixture.at(cltId)->getMean();
		for (int idx = 0; idx < nData; ++idx) {
			distP2M(idx, cltId) = globalGaussian->applyDistance(data.col(idx), mean);
		}
	}
}

} /* namespace dml */
