/*
 * MPCKMeans.cpp
 *
 *  Created on: Jun 24, 2015
 *      Author: vvminh
 */

#include "MPCKMeans.h"
#include "../utils/functionUtils.h"

#include <iostream>
#include <limits>

namespace dml {

using namespace Eigen;

MPCKMeans::MPCKMeans(const ConstMatrixRef& dataset, const int numClts,
	const std::string constraintFileName, const CovType type)
	:PCKMeans(dataset, numClts, constraintFileName, type) {}

MPCKMeans::~MPCKMeans() {}

/*virtual*/ void MPCKMeans::updateMixtures() {
	for (int cltId = 0; cltId < nClusters; ++cltId) {
		vMixture.at(cltId)->updateMean();
		vMixture.at(cltId)->updateConstraintImpact(data, vAssign, constr);
		vMixture.at(cltId)->cacheDistPoint2Point(data);
		vMixture.at(cltId)->cacheDistPoint2Mean(data);
		// vMixture.at(cltId)->debugCachedDistance();
	}
}

} /* namespace dml */
