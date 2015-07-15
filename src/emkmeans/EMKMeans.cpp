/*
 * EMBasicKMeans.cpp
 *
 *  Created on: Jun 4, 2015
 *      Author: vvminh
 */

#include "EMKMeans.h"

#include <cassert>
#include <cmath>
#include <limits>
#include <iostream>

#include "../utils/functionUtils.h"
#include "../gaussian/SimpleGaussian.cpp"
#include "../gaussian/DiagGaussian.cpp"
#include "../gaussian/FullGaussian.cpp"

namespace dml {

using namespace Eigen;

EMKMeans::EMKMeans(const Ref<const MatrixXf>& dataset, const int numClts, const CovType type) {
	data = dataset;
	nDims = data.rows();
	nData = data.cols();
	nClusters = numClts;
	covType = type;

	vAssign.reserve(nData);
	vAssign.assign(nData, 0);

	createMixtures();
}

void EMKMeans::createMixtures() {
	vMixture.reserve(nClusters);
	int nEstimateSize = nData / nClusters;
	for (int clusterId = 0; clusterId < nClusters; ++clusterId) {
		switch (covType) {
			case COV_NONE:
				vMixture.push_back( GaussianPtr(
					new SimpleGaussian(clusterId, nEstimateSize, nDims) ) );
			break;
			case COV_DIAG:
				vMixture.push_back( GaussianPtr(
					new DiagGaussian(clusterId, nEstimateSize, nDims) ) );
			break;
			case COV_FULL:
			vMixture.push_back( GaussianPtr(
					new FullGaussian(clusterId, nEstimateSize, nDims) ) );
			break;
			default:
				assert(false && "Invalid covariance type!");
			break;
		}
	}
}

float EMKMeans::getCurrentCost() {
	return ((0 == vObjFuncCached.size()) ? 0.0f : vObjFuncCached.back());
}

std::vector<int> EMKMeans::doClustering(const int maxIteration, const float minObjFuncChange) {
	maxIter = maxIteration;
	minChange = minObjFuncChange;
	//std::cout << "Do clustering with: nClusters=" << nClusters
			//<< ", maxIter=" << maxIter << ", minChange=" << minChange
			//<< ", nData=" << nData << ", nDims=" << nDims << '\n';
	createInitCenters();
	doVeryFirstClustering();
	runEM();
	return vAssign;
}

EMResult EMKMeans::getResult() {
    EMResult result;
    result.iterTerminate = currIter;
    result.cost = vObjFuncCached.back();
    float prevCost = vObjFuncCached.at(vObjFuncCached.size() - 1);
    result.reachLocalMinimal = (result.cost <= prevCost) ? 1.0f : 0.0f;
    return result;
}

void EMKMeans::createInitCenters() {
	int nEstimate = nData / nClusters;
	for (int cltId = 0; cltId < nClusters; ++cltId) {
		int rndIdx = cltId * nEstimate + randomInRange(0, nEstimate);
		vMixture.at(cltId)->setInitCenter(data.col(rndIdx));
	}
}

/*virtual*/ void EMKMeans::doVeryFirstClustering() {}

void EMKMeans::runEM()
{
	std::vector<int> randomIndex(nData);
	for (int i = 0; i < nData; ++i) {
		randomIndex.at(i) = i;
	}

	// start with a big value of objFunc, so the next iteration with be decreased
	vObjFuncCached.push_back(std::numeric_limits<float>::max());
	float currentCost = 0.0f;
	do {
		shuffleVector(randomIndex);
		runEStep(randomIndex);
		runMStep();
		currIter++;
		currentCost = calculateObjFunc();
		// std::cout << "@itr " << currIter
		// 		<< "\tcost = " << currentCost
		// 		<< "\tchange = " << vObjFuncCached.back() - currentCost
		// 		<< "\tdebugVMeasure = " << VMeasure(vAssign, nClusters, nClusters) << "\n\n";
	} while (false == checkConvergence(currentCost));
}

void EMKMeans::runEStep(const std::vector<int>& randomIndex) {
	findBestCluster(randomIndex);
	adaptMixturesSize();
	assignData();
}

void EMKMeans::adaptMixturesSize()
{
	std::vector<int> vCount(nClusters);
	vCount.assign(nClusters, 0);
	for (int i = 0; i < nData; ++i) {
		++vCount.at(vAssign.at(i));
	}
	// dump("vCount", vCount);

	for (int cltId = 0; cltId < nClusters; ++cltId) {
		vMixture.at(cltId)->adaptNewSize(vCount[cltId]);
	}
}

void EMKMeans::assignData()
{
	for (int i = 0; i < nData; ++i) {
		vMixture.at(vAssign.at(i))->insertDataPoint(data.col(i));
	}
}

void EMKMeans::runMStep()
{
	updateMixtures();
}

bool EMKMeans::checkConvergence(const float currentCost) {
	float prevCost = vObjFuncCached.back();
	vObjFuncCached.push_back(currentCost);
	bool convergenceWhenExceedMaxIterations = (currIter >= maxIter);
	bool convergenceWhenNoChange = (prevCost - currentCost <= minChange);
	return (convergenceWhenExceedMaxIterations || convergenceWhenNoChange);
}

} /* namespace dml */
