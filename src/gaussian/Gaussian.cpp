/*
 * Gaussian.cpp
 *
 *  Created on: Jun 19, 2015
 *      Author: vvminh
 */

#include "Gaussian.h"
#include <iostream>
#include <stdexcept>

namespace dml {

using namespace Eigen;

Gaussian::Gaussian(const int id, const int size, const int dimensions) {
	cltId = id;
	nSize = size;
	nDims = dimensions;

	mean = VectorXf(nDims);
	data = MatrixXf(nDims, nSize);
	adaptNewSize (nSize);
}

///////////////////////////////////////////////////////////////////////////////
// PUBLIC APIs
///////////////////////////////////////////////////////////////////////////////

void Gaussian::adaptNewSize(const int size) {
	nSize = size;
	nAssigned = 0;
	//assert((nSize > 0) && "EMPTY GAUSSIAN!");
	if (0 == nSize) {
		throw std::runtime_error("Can not create empty gaussian\n");
	}
	data.resize(nDims, nSize);
}

void Gaussian::insertDataPoint(const ConstVectorRef& dp) {
	assert((nAssigned < nSize) && "CAN NOT INSERT DATAPOINT TO GAUSSIAN");
	data.col(nAssigned) = dp;
	nAssigned++;
}

void Gaussian::setData(const ConstMatrixRef& X) {
	nAssigned = X.cols();
	nSize = X.cols();
	data = X;
}

float Gaussian::distance(const int idx1, const int idx2) {
	return distP2P(idx1, idx2);
}

float Gaussian::distanceToMean(const int idx) {
	return distP2M[idx];
}

/*virtual*/ void Gaussian::cacheDistPoint2Point(const ConstMatrixRef& Xp) {
	maxDist = 0.0f;
	if (0 == distP2P.cols() && 0 == distP2P.rows()) {
		distP2P = MatrixXf(Xp.cols(), Xp.cols());
	} else {
		distP2P.setZero();
	}

	for (int i = 0; i < Xp.cols() - 1; ++i) {
		for (int j = i; j < Xp.cols(); ++j) {
			const float dist = applyDistance(Xp.col(i), Xp.col(j));
			distP2P(i, j) = dist;
			distP2P(j, i) = dist;
			if (dist > maxDist) {
				maxDist = dist;
				farthest1 = i;
				farthest2 = j;
			}
		}
	}
}

/*virtual*/ void Gaussian::cacheDistPoint2Mean(const ConstMatrixRef& Xp) {
	if (0 == distP2M.size()) {
		distP2M = VectorXf(Xp.cols());
	} else {
		distP2M.setZero();
	}
	for (int i = 0; i < Xp.cols(); ++i) {
		distP2M[i] = applyDistance(Xp.col(i), mean);
	}
}

///////////////////////////////////////////////////////////////////////////////
// PUBLIC APIS - GETTER SETTER
///////////////////////////////////////////////////////////////////////////////

int Gaussian::getClusterId()
{
	return cltId;
}

float Gaussian::getMaxDistance()
{
	return maxDist;
}

float Gaussian::getLogDet()
{
	return logDet;
}

const Eigen::VectorXf Gaussian::getMean()
{
	return mean;
}

void Gaussian::setInitCenter(const ConstVectorRef& initCenter) {
	mean = initCenter;
}

///////////////////////////////////////////////////////////////////////////////
// PRIVATE METHODS
///////////////////////////////////////////////////////////////////////////////

void Gaussian::updateMean()
{
	mean = data.rowwise().mean();
}

void Gaussian::debugCachedDistance() {
	std::cout << "Debug cache point p2p: \n"  << distP2P.block(0, 0, 10, 10) << '\n';
	std::cout << "Debug cache point p2Mean: \n"  << distP2M.head(10).transpose() << '\n';
}

} /* namespace dml */
